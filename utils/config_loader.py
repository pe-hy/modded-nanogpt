import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader and accessor class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'dataset.train_examples')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_dict = self._config
        
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        config_dict[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """Save configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """Update configuration from command line arguments."""
        for key, value in args.items():
            if value is not None:
                self.set(key, value)
    
    @property
    def dataset(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._config.get('dataset', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def data_loader(self) -> Dict[str, Any]:
        """Get data loader configuration."""
        return self._config.get('data_loader', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get file paths configuration."""
        return self._config.get('paths', {})
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(config_path='{self.config_path}')"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)