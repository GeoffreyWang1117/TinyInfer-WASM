mod utils;
mod tensor;
mod ops;
mod engine;
mod error;

use wasm_bindgen::prelude::*;

#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    wasm_logger::init(wasm_logger::Config::default());
    log::info!("TinyInfer-WASM initialized");
}

/// Get version information
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check SIMD support
#[wasm_bindgen]
pub fn check_simd_support() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        // WASM SIMD is available if the module compiles with SIMD enabled
        cfg!(target_feature = "simd128")
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }
}
