import utils
from utils import check_local_file
from .clip import CustomCLIP


def build_encoder(config):
    vm_local_files_only, vm_file_name = check_local_file(config.pretrained_vision_name_or_path)
    with utils.transformers_log_level(40):  # 40 == logging.ERROR
        if config.encoder_type == "openai.clip":
            model = CustomCLIP.from_pretrained(
                vm_file_name,
                local_files_only=vm_local_files_only,
            )
        else:
            raise NotImplementedError()
    return model
