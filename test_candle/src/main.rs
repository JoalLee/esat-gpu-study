use candle_core::{Device, Tensor};

fn main() {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: metal={}", device.is_metal());
    
    if !device.is_metal() {
        println!("No Metal device available, exiting.");
        return;
    }
    
    // Test with real ESAT-sized data: W(50,3) * H(3,10) = WH(50,10)
    let w_data: Vec<f32> = (0..150).map(|i| (i as f32) * 0.01 + 0.5).collect();
    let h_data: Vec<f32> = (0..30).map(|i| (i as f32) * 0.01 + 0.1).collect();
    
    // Test CPU-then-device path
    let w_cpu = Tensor::from_vec(w_data.clone(), (50, 3), &Device::Cpu).unwrap();
    let w_metal = w_cpu.to_device(&device).unwrap();
    println!("W dtype: {:?}, device metal={}", w_metal.dtype(), w_metal.device().is_metal());
    
    let h_cpu = Tensor::from_vec(h_data.clone(), (3, 10), &Device::Cpu).unwrap();
    let h_metal = h_cpu.to_device(&device).unwrap();
    println!("H dtype: {:?}, device metal={}", h_metal.dtype(), h_metal.device().is_metal());
    
    match w_metal.matmul(&h_metal) {
        Ok(wh) => println!("CPU->Metal path matmul OK, shape={:?}", wh.shape()),
        Err(e) => println!("CPU->Metal path matmul FAILED: {}", e),
    }
    
    // Test direct on-device path
    let w_direct = Tensor::from_vec(w_data.clone(), (50, 3), &device).unwrap();
    let h_direct = Tensor::from_vec(h_data.clone(), (3, 10), &device).unwrap();
    match w_direct.matmul(&h_direct) {
        Ok(wh) => println!("Direct Metal matmul OK, shape={:?}", wh.shape()),
        Err(e) => println!("Direct Metal matmul FAILED: {}", e),
    }
}
