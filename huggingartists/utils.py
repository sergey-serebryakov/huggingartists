import copy
import json
import os
import typing as t
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "ParameterSource",
    "get_params",
    "get_path",
    "normalize_str",
    "default_workspace",
    "artist_workspace",
    "default_param_file",
    "load_mlcube_parameters",
    "update_attributes",
    "init_loggers",
    "init_environment",
]

ParameterSource = t.Union[t.Dict, str, Path]


def get_params(
    params: t.Optional[ParameterSource] = None, defaults: t.Optional[t.Dict] = None
) -> t.Dict:
    if params is None:
        params = {}
    if isinstance(params, (str, Path)):
        params = _load_from_file(params)
    if not isinstance(params, dict):
        raise NotImplementedError(f"Unsupported parameter source ({params}).")
    if defaults:
        _params = copy.deepcopy(defaults)
        _params.update(params)
        params = _params
    return params


def get_path(
    path: t.Optional[t.Union[str, Path]], default_path=t.Optional[t.Union[str, Path]]
) -> t.Optional[Path]:
    if path is not None:
        return _as_path(path)
    if default_path is not None:
        return _as_path(default_path)
    return None


def normalize_str(str_val: str) -> str:
    return "_".join(str_val.lower().strip().split())


def default_workspace() -> Path:
    return Path.cwd().resolve() / "workspace"


def default_param_file() -> str:
    return (Path.cwd() / "workspace" / "params.yaml").as_posix()


def artist_workspace(
    workspace_dir: t.Optional[t.Union[str, Path]], artist_name: str
) -> Path:
    return get_path(workspace_dir, default_workspace()) / normalize_str(artist_name)


def load_mlcube_parameters(param_file: t.Union[str, Path], task_name: str) -> t.Dict:
    _params: DictConfig = OmegaConf.load(param_file)
    assert isinstance(_params, DictConfig), f"Invalid param file ({param_file})."

    OmegaConf.resolve(_params)
    mlcube_params: t.Dict = OmegaConf.to_object(_params)
    assert isinstance(
        mlcube_params, dict
    ), f"Unexpected type ({type(mlcube_params)}). Should be dict."

    task_params = mlcube_params.get("base", None) or {}
    task_params.update(mlcube_params.get(task_name, None) or {})
    return task_params


def init_loggers(log_dir: t.Optional[t.Union[str, Path]] = None) -> None:
    import logging.config

    if log_dir is None:
        log_dir = Path.cwd()
    log_config = _load_from_file(Path(__file__).parent / "logger.yaml")
    log_config["handlers"]["file_handler"]["filename"] = (
        Path(log_dir) / log_config["handlers"]["file_handler"]["filename"]
    ).as_posix()
    logging.config.dictConfig(log_config)


def init_environment(params: t.Dict) -> None:
    for name, value in params.items():
        if name.startswith("env."):
            os.environ[name[4:]] = str(value)


def update_attributes(obj: t.Any, params: t.Dict, namespace: str) -> None:
    if not namespace.endswith("."):
        namespace = namespace + "."
    namespace_len = len(namespace)
    for name, value in params.items():
        if name.startswith(namespace):
            setattr(obj, name[namespace_len:], value)


def _load_from_file(path: t.Union[str, Path]) -> t.Dict:
    path = Path(path)
    if path.suffix in (".json", ".yaml"):
        with open(path, "rt") as fp:
            if path.suffix == ".yaml":
                params = yaml.load(fp, Loader=yaml.SafeLoader)
            else:
                params = json.load(fp)
        if not isinstance(params, dict):
            raise ValueError(
                f"Unsupported parameter file (path={path.as_posix()}, content_type={type(params)})."
            )
    else:
        raise NotImplementedError(
            f"Parameter source ({path.as_posix()}) not supported."
        )
    return params


def _as_path(path: t.Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        raise ValueError(f"Unsupported path object ({path}).")
    return path
