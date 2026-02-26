import re
from typing import Optional, Union

from mmengine.registry import VISBACKENDS
from mmengine.visualization import MLflowVisBackend

@VISBACKENDS.register_module()
class SanitizedMLflowVisBackend(MLflowVisBackend):
    """Custom MLflow visualization backend that sanitizes metric names.
    
    MLflow only allows alphanumerics, underscores (_), dashes (-), periods (.), 
    spaces ( ), colon(:) and slashes (/).
    COCO metrics often contain parentheses like 'coco/AP (M)', which results in error.
    """

    def _init_env(self):
        super()._init_env()

    def _sanitize_name(self, name: str) -> str:
        # Replace any character not in [a-zA-Z0-9_./: -] with '_'
        # Actually MLflow allows spaces, but underscores are safer and cleaner
        sanitized = re.sub(r'[^a-zA-Z0-9_./:-]', '_', name)
        # Replace multiple underscores with a single one
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        sanitized_name = self._sanitize_name(name)
        super().add_scalar(sanitized_name, value, step, **kwargs)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        sanitized_dict = {self._sanitize_name(k): v for k, v in scalar_dict.items()}
        super().add_scalars(sanitized_dict, step, file_path, **kwargs)
