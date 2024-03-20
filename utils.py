import os


def check_upload_dirs(*dirs):
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            raise ValueError(f'файл {dir_path} не существует ((')
        
def check_dump_dirs(*dirs):
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            try:
                os.mkdir(
                    '/'.join(
                        dir_path.split('/')[:-1]
                    )
                )
            except:
                print(f'Путь, по которому лежит файл, уже существует')
