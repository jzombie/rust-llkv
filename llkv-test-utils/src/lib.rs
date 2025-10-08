use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize tracing for test binaries. Safe to call multiple times.
pub fn init_tracing_for_tests() {
    INIT.call_once(|| {
        use tracing_subscriber::filter::EnvFilter;
        use tracing_subscriber::fmt;
        let env = std::env::var("RUST_LOG").ok();
        let filter = match env {
            Some(_) => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            None => EnvFilter::new("info"),
        };
        fmt().with_env_filter(filter).with_target(false).init();
    });
}

#[cfg(feature = "auto-init")]
mod auto {
    // Use ctor to run at binary init time to avoid having to call init in every test.
    use ctor::ctor;

    #[ctor]
    fn init() {
        super::init_tracing_for_tests();
    }
}
