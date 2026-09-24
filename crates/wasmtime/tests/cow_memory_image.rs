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
                memory.grow(&mut store, 1)?;
                assert!(
                    memory.data(&store)[2 * 65536..]
                        .iter()
                        .all(|byte| *byte == 0),
                    "growth after restore must expose zeroed pages"
                );
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
    fn a_guest_reading_a_truncated_image_traps() -> Result<()> {
        let engine = engine(false, 0)?;
        let module = Module::new(
            &engine,
            r#"(module
            (memory (export "memory") 1 4)
            (func (export "read") (param i32) (result i32)
                local.get 0 i32.load8_u))"#,
        )?;
        let mut file = tempfile::tempfile()?;
        file.write_all(&[42; 65536])?;
        let backing = file.try_clone()?;
        // SAFETY: the test breaks this contract on purpose, only while the
        // guest alone reads the image, to show that guest reads are contained.
        let snapshot = unsafe { CowMemoryImage::from_file(file, 0, 65536)? };
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();
        // SAFETY: the instance is idle and has no mutable state outside this memory.
        unsafe { memory.restore_cow_image(&mut store, &snapshot)? };
        backing.set_len(0)?;
        let read = instance.get_typed_func::<i32, i32>(&mut store, "read")?;
        let error = read
            .call(&mut store, 0)
            .expect_err("a guest must not read storage that no longer exists");
        assert_eq!(
            error.downcast_ref::<Trap>(),
            Some(&Trap::MappedStorageFault),
            "failed image storage must surface as this guest's trap: {error:?}"
        );
        Ok(())
    }

    fn filled(pages: &[u8]) -> Result<CowMemorySource> {
        let mut file = tempfile::tempfile()?;
        for byte in pages {
            file.write_all(&vec![*byte; 65536])?;
        }
        // SAFETY: nothing writes the file after it is retained.
        unsafe { CowMemorySource::from_file(file) }
    }

    fn extent(source: usize, file_page: u64, memory_page: usize) -> CowMemorySourceExtent {
        CowMemorySourceExtent {
            source,
            file_offset: file_page * 65536,
            memory_offset: memory_page * 65536,
            length: 65536,
        }
    }

    fn restored(engine: &Engine, image: &CowMemoryImage) -> Result<(Store<()>, Memory)> {
        let module = Module::new(engine, r#"(module (memory (export "memory") 1 4))"#)?;
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();
        memory.grow(&mut store, 1)?;
        // SAFETY: the instance is idle and has no mutable state outside this memory.
        unsafe { memory.restore_cow_image(&mut store, image)? };
        Ok((store, memory))
    }

    #[test]
    fn an_image_maps_each_extent_from_its_own_source() -> Result<()> {
        for pool in [false, true] {
            let engine = engine(pool, 0)?;
            let base = filled(&[0x11, 0x22])?;
            let layer = filled(&[0x33])?;
            let image = CowMemoryImage::from_sources(
                &[base, layer],
                2 * 65536,
                vec![extent(0, 0, 0), extent(1, 0, 1)],
            )?;
            for _ in 0..2 {
                let (mut store, memory) = restored(&engine, &image)?;
                assert_eq!(
                    (memory.data(&store)[0], memory.data(&store)[65536]),
                    (0x11, 0x33),
                    "each extent must map from the source it names"
                );
                memory.data_mut(&mut store)[65536] = 0x44;
            }
        }
        Ok(())
    }

    #[test]
    fn images_sharing_a_source_keep_their_own_extents() -> Result<()> {
        let engine = engine(false, 0)?;
        let base = filled(&[0x11, 0x22])?;
        let first = CowMemoryImage::from_sources(
            &[base.clone(), filled(&[0x33])?],
            2 * 65536,
            vec![extent(0, 0, 0), extent(1, 0, 1)],
        )?;
        let second = CowMemoryImage::from_sources(
            &[base, filled(&[0x55])?],
            2 * 65536,
            vec![extent(1, 0, 0), extent(0, 1, 1)],
        )?;
        let (first_store, first_memory) = restored(&engine, &first)?;
        let (second_store, second_memory) = restored(&engine, &second)?;
        assert_eq!(
            (
                first_memory.data(&first_store)[0],
                first_memory.data(&first_store)[65536]
            ),
            (0x11, 0x33),
            "the first image must keep its own layout"
        );
        assert_eq!(
            (
                second_memory.data(&second_store)[0],
                second_memory.data(&second_store)[65536]
            ),
            (0x55, 0x22),
            "a second image over the same source must keep its own layout"
        );
        Ok(())
    }

    #[test]
    fn rejects_extents_naming_a_missing_or_overrun_source() -> Result<()> {
        let sources = [filled(&[0x11])?, filled(&[0x22])?];
        for extents in [
            vec![extent(0, 0, 0), extent(2, 0, 1)],
            vec![extent(0, 0, 0), extent(1, 1, 1)],
        ] {
            assert!(
                CowMemoryImage::from_sources(&sources, 2 * 65536, extents.clone()).is_err(),
                "an extent outside its sources must be refused: {extents:?}"
            );
        }
        assert!(
            CowMemoryImage::from_sources(&[], 65536, vec![extent(0, 0, 0)]).is_err(),
            "an image needs at least one source"
        );
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
    fn rejects_malformed_extent_layouts() -> Result<()> {
        let extent = |file_offset, memory_offset, length| CowMemoryExtent {
            file_offset,
            memory_offset,
            length,
        };
        let cases = [
            vec![],
            vec![extent(0, 0, 0)],
            vec![extent(1, 0, 65536)],
            vec![extent(0, 1, 65536)],
            vec![extent(0, 0, 1)],
            vec![extent(65536, 0, 131072)],
            vec![extent(0, 0, 65536), extent(0, 0, 65536)],
            vec![extent(0, 0, 65536), extent(0, 131072, 65536)],
            vec![extent(0, 0, 65536)],
            vec![extent(u64::MAX - 4095, 0, 131072)],
        ];
        for extents in cases {
            let file = tempfile::tempfile()?;
            file.set_len(131072)?;
            // SAFETY: no other file handle exists and no writer can mutate the image.
            assert!(
                unsafe { CowMemoryImage::from_file_extents(file, 131072, extents.clone()) }
                    .is_err(),
                "malformed layout must be refused: {extents:?}"
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

    fn exercise_without_initial_image(pool: bool) -> Result<()> {
        let engine = engine(pool, 0)?;
        let module = Module::new(&engine, "(module (memory (export \"memory\") 1 4))")?;
        let (snapshot, _) = image()?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance.get_memory(&mut store, "memory").unwrap();
        memory.grow(&mut store, 1)?;
        // SAFETY: the instance is idle and has no mutable state outside this memory.
        unsafe { memory.restore_cow_image(&mut store, &snapshot)? };
        assert_eq!(
            memory.data(&store)[0],
            42,
            "a memory whose module has no data segments must still receive a snapshot"
        );
        assert_eq!(
            memory.data(&store)[2 * 65536 - 1],
            73,
            "the whole snapshot must be visible"
        );
        memory.grow(&mut store, 1)?;
        assert!(
            memory.data(&store)[2 * 65536..]
                .iter()
                .all(|byte| *byte == 0),
            "growth after restore must expose zeroed pages"
        );
        Ok(())
    }

    #[test]
    fn on_demand_memories_without_an_initial_image_accept_snapshots() -> Result<()> {
        exercise_without_initial_image(false)
    }
    #[test]
    fn pooled_memories_without_an_initial_image_accept_snapshots() -> Result<()> {
        exercise_without_initial_image(true)
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
