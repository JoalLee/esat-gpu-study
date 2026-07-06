use candle_core::Device;

/// Try to initialize Metal device; return None and warn on panic/error.
#[cfg(not(feature = "cuda"))]
fn try_new_metal() -> Option<Device> {
    match std::panic::catch_unwind(|| Device::new_metal(0)) {
        Ok(Ok(dev)) => {
            eprintln!("[esat_rust] Metal device initialized successfully");
            Some(dev)
        }
        Ok(Err(e)) => {
            eprintln!("[esat_rust] Warning: Metal init failed ({}), falling back to CPU", e);
            None
        }
        Err(panic_info) => {
            let msg = panic_info
                .downcast_ref::<&str>().map(|s| s.to_string())
                .or_else(|| panic_info.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            eprintln!("[esat_rust] Warning: Metal init panicked ({}), falling back to CPU", msg);
            None
        }
    }
}

pub fn select_device(prefer_gpu: bool) -> (Device, &'static str) {
    if !prefer_gpu {
        return (Device::Cpu, "cpu");
    }

    #[cfg(feature = "cuda")]
    {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let backend = if device.is_cpu() { "cpu" } else { "cuda" };
        return (device, backend);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let device = try_new_metal().unwrap_or(Device::Cpu);
        let backend = if device.is_cpu() { "cpu" } else { "metal" };
        (device, backend)
    }
}
