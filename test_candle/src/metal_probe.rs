use candle_metal_kernels::metal::Device;
use objc2_metal::MTLCreateSystemDefaultDevice;

fn main() {
    let raw_default = MTLCreateSystemDefaultDevice();
    println!("MTLCreateSystemDefaultDevice: {}", raw_default.is_some());

    let all = Device::all();
    println!("Device::all len: {}", all.len());
    for (i, dev) in all.iter().enumerate() {
        println!("  all[{i}] registry_id={}", dev.registry_id());
    }

    let default = Device::system_default();
    println!("Device::system_default: {}", default.is_some());
    if let Some(dev) = default {
        println!("  default registry_id={}", dev.registry_id());
    }
}
