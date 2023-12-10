import yaml
import os


def load_config(config_filename='config_default.yaml'):
    # 构建 config.yaml 的路径
    # config_path = os.path.join(os.path.dirname(__file__), '..', config_filename)

    # 读取配置文件
    with open("config/" + config_filename, 'r') as file:
        config = yaml.safe_load(file)

    return config


def update_config(config, config_filename='config.yaml'):
    # 构建 config.yaml 的路径
    # 假设您的工作目录是项目的根目录
    config_path = os.path.join(config['project_dir'], 'config', config_filename)

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    print("config写入成功")


def update_project_dir(config_filename):
    config = load_config(config_filename)
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['project_dir'] = project_directory
    update_config(config,config_filename)
    return config


if __name__ == '__main__':
    config = load_config()
    project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['project_dir'] = project_directory
    update_config(config)
