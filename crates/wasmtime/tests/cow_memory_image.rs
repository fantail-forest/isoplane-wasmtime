#[cfg(all(
    target_os = "linux",
    feature = "std",
    feature = "cranelift",
    feature = "pooling-allocator",
    feature = "wat"
))]
mod tests {
    use std::io::{Seek, SeekFrom, Write};
    use wasmtime::*;

    fn engine(pool: bool, resident: usize) -> Result<Engine> {
        let mut config = Config::new();
        config
            .memory_init_cow(true)
            .memory_reservation(4 * 65536)
            .memory_guard_size(65536);
        if pool {
            let mut allocation = PoolingAllocationConfig::default();
            allocation
                .total_memories(1)
                .total_core_instances(1)
                .max_memory_size(4 * 65536)
                .linear_memory_keep_resident(resident);
            config.allocation_strategy(InstanceAllocationStrategy::Pooling(allocation));
        }
        Engine::new(&config)
    }

    fn image() -> Result<(CowMemoryImage, u64)> {
        let mut file = tempfile::tempfile()?;
        file.write_all(&vec![99; 65536])?;
        let mut bytes = vec![0; 2 * 65536];
        bytes[0] = 42;
        bytes[2 * 65536 - 1] = 73;
        file.write_all(&bytes[65536..])?;
        file.write_all(&bytes[..65536])?;
        file.seek(SeekFrom::Start(0))?;
        // SAFETY: no writer survives construction; the retained inode never changes.
        use std::os::unix::fs::MetadataExt;
        let inode = file.metadata()?.ino();
        Ok((
            unsafe {
                CowMemoryImage::from_file_extents(
                    file,
                    2 * 65536,
                    vec![
                        CowMemoryExtent {
                            file_offset: 2 * 65536,
                            memory_offset: 0,
                            length: 65536,
                        },
                        CowMemoryExtent {
                            file_offset: 65536,
                            memory_offset: 65536,
                            length: 65536,
                        },
                    ],
                )?
            },
            inode,
        ))
    }

    fn exercise(pool: bool, resident: usize) -> Result<()> {
        let engine = engine(pool, resident)?;
        let module = Module::new(
            &engine,
            r#"(module
            (memory (export "memory") 1 4)
            (data (i32.const 0) "\11")
            (func (export "read") (param i32) (result i32)
                local.get 0 i32.load8_u))"#,
        )?;
        let (snapshot, inode) = image()?;
        for _ in 0..4 {
            {
                let mut store = Store::new(&engine, ());
                let instance = Instance::new(&mut store, &module, &[])?;
                let memory = instance.get_memory(&mut store, "memory").unwrap();
                assert_eq!(
                    memory.data(&store)[0],
                    17,
                    "slot reuse must restore the compiled baseline"
                );
                memory.grow(&mut store, 1)?;
                // SAFETY: no guest call is active and this module has no other mutable state.
                // SAFETY: the instance is idle and has no mutable state outside this memory.
                unsafe { memory.restore_cow_image(&mut store, &snapshot)? };
                let pointer = memory.data_ptr(&store) as usize;
                let maps = std::fs::read_to_string("/proc/self/maps")?;
                let fields = maps
                    .lines()
                    .find_map(|line| {
                        let fields: Vec<_> = line.split_whitespace().collect();
                        let (start, end) = fields[0].split_once('-')?;
                        let start = usize::from_str_radix(start, 16).ok()?;
                        let end = usize::from_str_radix(end, 16).ok()?;
                        (start <= pointer && pointer < end).then_some(fields)
                    })
                    .expect("memory mapping");
                assert_eq!(
                    fields[1], "rw-p",
                    "snapshot memory must use private writable mappings"
                );
                assert_eq!(
                    fields[4].parse::<u64>()?,
                    inode,
                    "memory must map the snapshot's actual file inode"
                );
                assert_eq!(
                    memory.data(&store)[0],
                    42,
                    "the layer must replace the original byte"
                );
                assert_eq!(
                    memory.data(&store)[2 * 65536 - 1],
                    73,
                    "the offset image must cover its last page"
                );
                memory.data_mut(&mut store)[0] = 88;
            }
            {
                let mut store = Store::new(&engine, ());
                let instance = Instance::new(&mut store, &module, &[])?;
                let memory = instance.get_memory(&mut store, "memory").unwrap();
                assert_eq!(
                    memory.size(&store),
                    1,
                    "reused slots must regain their original logical size"
                );
                let read = instance.get_typed_func::<i32, i32>(&mut store, "read")?;
                assert!(
                    read.call(&mut store, 65536).is_err(),
                    "reused slot must restore its guard boundary"
                );
                memory.grow(&mut store, 1)?;
                // SAFETY: the instance is idle and has no mutable state outside this memory.
                unsafe { memory.restore_cow_image(&mut store, &snapshot)? };
                assert_eq!(
                    memory.data(&store)[0],
                    42,
                    "private writes must not change a later activation"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_image_ranges() -> Result<()> {
        for (offset, length) in [(1, 65536), (0, 1), (65536, 65536), (0, 0)] {
            let file = tempfile::tempfile()?;
            file.set_len(65536)?;
            // SAFETY: this file has no other writer and is never modified afterward.
            assert!(
                unsafe { CowMemoryImage::from_file(file, offset, length) }.is_err(),
                "invalid image offset/length must be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_image_larger_than_current_memory() -> Result<()> {
        let engine = engine(true, 0)?;
        let module = Module::new(&engine, "(module (memory (export \"memory\") 1 4))")?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();
        let (snapshot, _) = image()?;
        // SAFETY: the instance is idle and has no mutable state outside this memory.
        assert!(
            unsafe { memory.restore_cow_image(&mut store, &snapshot) }.is_err(),
            "restoration must not bypass growth limits"
        );
        Ok(())
    }

    #[test]
    fn pooled_decommitted_slots_preserve_isolation() -> Result<()> {
        exercise(true, 0)
    }
    #[test]
    fn pooled_resident_slots_preserve_isolation() -> Result<()> {
        exercise(true, 4 * 65536)
    }
    #[test]
    fn on_demand_slots_preserve_isolation() -> Result<()> {
        exercise(false, 0)
    }
}
