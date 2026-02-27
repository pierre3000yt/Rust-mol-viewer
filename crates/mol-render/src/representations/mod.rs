pub mod spheres;
pub mod ball_stick;
pub mod billboards;
pub mod ribbon;
pub mod surface;
pub mod surface_raymarch;
pub mod sphere_raymarch;
pub mod axes;

pub use spheres::SpheresRenderer;
pub use ball_stick::BallStickRenderer;
pub use billboards::BillboardRenderer;
pub use ribbon::RibbonRenderer;
pub use surface::{SurfaceRenderer, SurfaceConfig};
pub use surface_raymarch::{SurfaceRaymarchRenderer, RaymarchConfig};
pub use sphere_raymarch::SphereRaymarchRenderer;
pub use axes::AxesRenderer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationType {
    VanDerWaals,
    BallAndStick,
    Ribbon,
    Surface,
    SurfaceRaymarch,    // Real-time raymarching SDF surface (slow)
    SphereRaymarch,     // Raymarched individual spheres (fast)
}
