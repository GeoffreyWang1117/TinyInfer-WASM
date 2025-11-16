use wasm_bindgen::prelude::*;

/// Set panic hook for better error messages
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Log to browser console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn error(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn warn(s: &str);
}

/// Macro for console.log
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        $crate::utils::log(&format_args!($($t)*).to_string())
    }
}

/// Get current timestamp in milliseconds
#[wasm_bindgen]
pub fn now() -> f64 {
    js_sys::Date::now()
}

/// Performance timer
pub struct Timer {
    start: f64,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: now(),
        }
    }

    pub fn elapsed(&self) -> f64 {
        now() - self.start
    }

    pub fn reset(&mut self) {
        self.start = now();
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed() >= 10.0);
    }
}
