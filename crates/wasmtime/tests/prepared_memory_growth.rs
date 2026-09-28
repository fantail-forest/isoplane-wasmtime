#![cfg(all(
    target_os = "linux",
    feature = "std",
    feature = "cranelift",
    feature = "pooling-allocator",
    feature = "wat"
))]

use wasmtime::*;

fn engine(pooling: bool, explicit: bool) -> Result<Engine> {
    let mut config = Config::new();
    config
        .memory_init_cow(true)
        .memory_reservation(4 * 65536)
        .memory_guard_size(65536)
        .memory_may_move(false)
        .memory_explicit_bounds_checks(explicit);
    if pooling {
        let mut pool = PoolingAllocationConfig::default();
        pool.total_memories(1)
            .total_core_instances(1)
            .max_memory_size(4 * 65536);
        config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));
    }
    Engine::new(&config)
}

const MODULE: &str = r#"(module
  (memory (export "memory") 1 4)
  (func (export "read") (param i32) (result i32) local.get 0 i32.load8_u)
  (func (export "offset") (param i32) (result i32) local.get 0 i32.load8_u offset=65536)
  (func (export "wide") (param i32) (result i64) local.get 0 i64.load)
  (func (export "write") (param i32) local.get 0 i32.const 42 i32.store8)
  (func (export "fill") (param i32) local.get 0 i32.const 42 i32.const 8 memory.fill)
  (func (export "grow") (param i32) (result i32) local.get 0 memory.grow)
)"#;

#[test]
fn accessible_pages_remain_out_of_bounds_until_logical_growth() -> Result<()> {
    for pooling in [false, true] {
        let engine = engine(pooling, true)?;
        let module = Module::new(&engine, MODULE)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .expect("memory export");
        memory.prepare_growth(&mut store, 4 * 65536)?;
        assert_eq!(
            memory.size(&store),
            1,
            "permissions must not change logical size"
        );
        assert_eq!(
            memory.data_size(&store),
            65536,
            "host access remains logically bounded"
        );
        for (name, address) in [("read", 65536), ("offset", 0)] {
            let f = instance.get_typed_func::<i32, i32>(&mut store, name)?;
            let e = f
                .call(&mut store, address)
                .expect_err("accessible suffix must trap");
            assert_eq!(
                e.downcast_ref::<Trap>(),
                Some(&Trap::MemoryOutOfBounds),
                "logical bounds must trap"
            );
        }
        let wide = instance.get_typed_func::<i32, i64>(&mut store, "wide")?;
        assert_eq!(
            wide.call(&mut store, 65532)
                .expect_err("cross-boundary load")
                .downcast_ref::<Trap>(),
            Some(&Trap::MemoryOutOfBounds),
            "access width must be checked"
        );
        for (name, address) in [("write", 65536), ("fill", 65532)] {
            let f = instance.get_typed_func::<i32, ()>(&mut store, name)?;
            assert_eq!(
                f.call(&mut store, address)
                    .expect_err("suffix write")
                    .downcast_ref::<Trap>(),
                Some(&Trap::MemoryOutOfBounds),
                "writes must respect logical bounds"
            );
        }
        let grow = instance.get_typed_func::<i32, i32>(&mut store, "grow")?;
        assert_eq!(
            grow.call(&mut store, 1)?,
            1,
            "growth must report the original logical size"
        );
        let read = instance.get_typed_func::<i32, i32>(&mut store, "read")?;
        assert_eq!(
            read.call(&mut store, 65536)?,
            0,
            "newly exposed bytes must be zero"
        );
    }
    Ok(())
}

#[test]
fn preparing_growth_requires_compatible_compiled_bounds_checks() -> Result<()> {
    let engine = engine(true, false)?;
    let module = Module::new(&engine, MODULE)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");
    assert!(
        memory.prepare_growth(&mut store, 4 * 65536).is_err(),
        "signal-based bounds must reject accessible suffixes"
    );
    let checked_engine = self::engine(true, true)?;
    let bytes = module.serialize()?;
    // SAFETY: bytes were serialized above by this Wasmtime build.
    assert!(
        unsafe { Module::deserialize(&checked_engine, bytes) }.is_err(),
        "cached code compiled with incompatible bounds checks must be rejected"
    );
    Ok(())
}

#[test]
fn prepared_growth_still_obeys_the_store_limit() -> Result<()> {
    let engine = engine(true, true)?;
    let module = Module::new(&engine, MODULE)?;
    let mut store = Store::new(
        &engine,
        StoreLimitsBuilder::new().memory_size(2 * 65536).build(),
    );
    store.limiter(|limits| limits);
    let instance = Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");
    memory.prepare_growth(&mut store, 4 * 65536)?;
    let grow = instance.get_typed_func::<i32, i32>(&mut store, "grow")?;
    assert_eq!(grow.call(&mut store, 1)?, 1, "allowed growth must succeed");
    assert_eq!(
        grow.call(&mut store, 1)?,
        -1,
        "accessible pages must not bypass the store limit"
    );
    assert_eq!(
        memory.size(&store),
        2,
        "refused growth must preserve logical size"
    );
    assert!(
        memory.prepare_growth(&mut store, 5 * 65536).is_err(),
        "preparation must stay inside its reservation"
    );
    Ok(())
}

#[test]
fn prepared_pages_are_demand_allocated_and_reset_on_pool_reuse() -> Result<()> {
    use std::os::unix::fs::FileExt;
    let engine = engine(true, true)?;
    let module = Module::new(&engine, MODULE)?;
    for _ in 0..3 {
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .expect("memory export");
        memory.prepare_growth(&mut store, 4 * 65536)?;
        let address = memory.data_ptr(&store) as u64;
        let mut entries = vec![0; 3 * 65536 / 4096 * 8];
        std::fs::File::open("/proc/self/pagemap")?
            .read_exact_at(&mut entries, ((address + 65536) / 4096) * 8)?;
        assert!(
            entries.chunks_exact(8).all(|entry| u64::from_ne_bytes(
                entry.try_into().expect("entry width")
            ) & (1 << 63)
                == 0),
            "accessible suffix must have no resident pages before use"
        );
        memory.grow(&mut store, 3)?;
        assert!(
            memory.data(&store)[65536..].iter().all(|b| *b == 0),
            "pool reuse must clear previous private writes"
        );
        memory.data_mut(&mut store)[65536..].fill(73);
    }
    Ok(())
}

#[test]
fn preparation_rejects_a_module_ceiling_smaller_than_the_reservation() -> Result<()> {
    let engine = engine(true, true)?;
    let module = Module::new(&engine, "(module (memory (export \"memory\") 1 2))")?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");
    assert!(
        memory.prepare_growth(&mut store, 3 * 65536).is_err(),
        "reservation capacity must not override the module maximum"
    );
    assert_eq!(
        memory.size(&store),
        1,
        "failed preparation must preserve logical size"
    );
    Ok(())
}

#[test]
fn preparation_rejects_memory_without_a_cow_slot() -> Result<()> {
    let mut config = Config::new();
    config
        .memory_explicit_bounds_checks(true)
        .memory_init_cow(false)
        .memory_reservation(4 * 65536);
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(1, Some(4)))?;
    assert!(
        memory.prepare_growth(&mut store, 4 * 65536).is_err(),
        "an unsupported allocator must reject preparation"
    );
    Ok(())
}
