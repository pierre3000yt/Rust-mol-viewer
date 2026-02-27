use mol_render::RepresentationType;
use pdb_parser::Trajectory;

/// Represents a loaded molecular model in the scene
#[derive(Debug)]
pub struct LoadedModel {
    /// Unique identifier for this model
    pub id: usize,
    /// Display name (typically filename without extension)
    pub name: String,
    /// Full file path
    pub path: String,
    /// The molecular data (trajectory with topology and frames)
    pub trajectory: Trajectory,
    /// Whether this model is currently visible in the scene
    pub visible: bool,
    /// Rendering representation type for this model
    pub representation: RepresentationType,
}

impl LoadedModel {
    /// Create a new loaded model
    pub fn new(
        id: usize,
        name: String,
        path: String,
        trajectory: Trajectory,
    ) -> Self {
        Self {
            id,
            name,
            path,
            trajectory,
            visible: true, // Visible by default
            representation: RepresentationType::VanDerWaals, // Default representation
        }
    }
}

/// Manages multiple loaded molecular models
#[derive(Debug, Default)]
pub struct ModelManager {
    models: Vec<LoadedModel>,
    next_id: usize,
}

impl ModelManager {
    /// Create a new empty model manager
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a new model to the manager
    ///
    /// Returns the unique ID assigned to this model
    pub fn add_model(
        &mut self,
        name: String,
        path: String,
        trajectory: Trajectory,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let model = LoadedModel::new(id, name, path, trajectory);
        self.models.push(model);

        log::info!("Added model {} with ID {}", self.models.last().unwrap().name, id);

        id
    }

    /// Remove a model by ID
    ///
    /// Returns the removed model if it existed
    pub fn remove_model(&mut self, id: usize) -> Option<LoadedModel> {
        if let Some(index) = self.models.iter().position(|m| m.id == id) {
            Some(self.models.remove(index))
        } else {
            None
        }
    }

    /// Get a reference to a model by ID
    pub fn get_model(&self, id: usize) -> Option<&LoadedModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Get a mutable reference to a model by ID
    pub fn get_model_mut(&mut self, id: usize) -> Option<&mut LoadedModel> {
        self.models.iter_mut().find(|m| m.id == id)
    }

    /// Get an iterator over all visible models
    pub fn visible_models(&self) -> impl Iterator<Item = &LoadedModel> {
        self.models.iter().filter(|m| m.visible)
    }

    /// Get the total number of loaded models (visible + hidden)
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get the number of visible models
    pub fn visible_count(&self) -> usize {
        self.models.iter().filter(|m| m.visible).count()
    }

    /// Clear all models
    pub fn clear_all(&mut self) {
        log::info!("Clearing all models (count: {})", self.models.len());
        self.models.clear();
    }

    /// Check if any model is loaded
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Get an iterator over all models (visible and hidden)
    pub fn all_models(&self) -> impl Iterator<Item = &LoadedModel> {
        self.models.iter()
    }

    /// Get a mutable iterator over all models
    pub fn all_models_mut(&mut self) -> impl Iterator<Item = &mut LoadedModel> {
        self.models.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdb_parser::Protein;

    fn create_dummy_trajectory() -> Trajectory {
        Trajectory {
            topology: Protein {
                atoms: Vec::new(),
                chains: Vec::new(),
                bonds: Vec::new(),
                secondary_structure: pdb_parser::SecondaryStructure {
                    helices: Vec::new(),
                    sheets: Vec::new(),
                },
                octree: None,
            },
            frames: Vec::new(),
        }
    }

    #[test]
    fn test_model_manager_add() {
        let mut mgr = ModelManager::new();
        let traj = create_dummy_trajectory();
        let id = mgr.add_model("test".into(), "path".into(), traj);

        assert_eq!(mgr.model_count(), 1);
        assert_eq!(mgr.get_model(id).unwrap().id, id);
        assert_eq!(mgr.get_model(id).unwrap().name, "test");
    }

    #[test]
    fn test_model_manager_remove() {
        let mut mgr = ModelManager::new();
        let traj = create_dummy_trajectory();
        let id = mgr.add_model("test".into(), "path".into(), traj);

        let removed = mgr.remove_model(id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, id);
        assert_eq!(mgr.model_count(), 0);
    }

    #[test]
    fn test_visible_models_filter() {
        let mut mgr = ModelManager::new();
        let traj1 = create_dummy_trajectory();
        let traj2 = create_dummy_trajectory();

        let id1 = mgr.add_model("test1".into(), "path1".into(), traj1);
        let _id2 = mgr.add_model("test2".into(), "path2".into(), traj2);

        mgr.get_model_mut(id1).unwrap().visible = false;

        assert_eq!(mgr.model_count(), 2);
        assert_eq!(mgr.visible_count(), 1);
        assert_eq!(mgr.visible_models().count(), 1);
    }

    #[test]
    fn test_clear_all() {
        let mut mgr = ModelManager::new();
        mgr.add_model("test1".into(), "path1".into(), create_dummy_trajectory());
        mgr.add_model("test2".into(), "path2".into(), create_dummy_trajectory());

        assert_eq!(mgr.model_count(), 2);
        mgr.clear_all();
        assert_eq!(mgr.model_count(), 0);
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_unique_ids() {
        let mut mgr = ModelManager::new();
        let id1 = mgr.add_model("test1".into(), "path1".into(), create_dummy_trajectory());
        let id2 = mgr.add_model("test2".into(), "path2".into(), create_dummy_trajectory());
        let id3 = mgr.add_model("test3".into(), "path3".into(), create_dummy_trajectory());

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }
}
