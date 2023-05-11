from dotmap import DotMap
import toml


def get_config():
    with open("config.toml") as file:
        config = toml.load(file)
    return DotMap(config)
