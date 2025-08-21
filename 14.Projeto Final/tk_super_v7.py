import os
import re

# Pastas e arquivos a serem ignorados
IGNORED_FOLDERS = {
    '.venv', 'venv', 'env', '.idea', '.git', 'node_modules', '__pycache__',
    'dist', 'build', '.DS_Store', '.vscode', 'target', 'out',
    '.pytest_cache', '.mypy_cache', 'logs', 'coverage', '.css'
}

# Função para verificar se um item (pasta ou arquivo) deve ser ignorado
def should_ignore(item, full_path, exclude_files):
    return (
        item.startswith('.') or
        item in exclude_files or
        any(ignored in full_path for ignored in IGNORED_FOLDERS)
    )

# Função para identificar arquivos especiais .ts
def is_special_ts_file(filename):
    patterns = [
        r'\.schema\.ts$',
        r'\.service\.ts$',
        r'\.controller\.ts$',
        r'\.route\.ts$',  # Corrigido de .routes.ts para .route.ts
        r'\.routes\.ts$',  # Corrigido de .routes.ts para .route.ts
        r'\.d\.ts$',
        r'\.test\.ts$',
        r'\.config\.ts$',
        r'\.config\.mjs$',
        r'\.\([^)]+\)\.ts$',
        r'\.test\.ts$',
        r'\.spec\.ts$',
        r'\.config\.ts$',
        r'\.setup\.ts$',
        r'\.mock\.ts$',
        r'\.model\.ts$',
        r'\.middleware\.ts$',
        r'\.helper\.ts$',
        r'\.util\.ts$',
        r'\.factory\.ts$',
        r'\.validator\.ts$',
        r'\.env\.ts$',
        r'\.main\.ts$',
        r'\.init\.ts$',
        r'\.web\.ts$'
    ]
    for pattern in patterns:
        if re.search(pattern, filename):
            return True
    return False

# Função para identificar arquivos especiais de Python
def is_special_py_file(filename):
    patterns = [
        r'_test\.py$',         # test_xxx.py
        r'^test_.*\.py$',      # test_xxx.py
        r'_spec\.py$',         # xxx_spec.py
        r'_config\.py$',       # xxx_config.py
        r'_settings\.py$',     # xxx_settings.py
        r'_schema\.py$',       # xxx_schema.py
        r'_model\.py$',        # xxx_model.py
        r'_views?\.py$',       # xxx_view.py ou xxx_views.py
        r'_forms?\.py$',       # xxx_form.py ou xxx_forms.py
        r'_admin\.py$',        # xxx_admin.py
        r'_serializers?\.py$', # xxx_serializer.py ou xxx_serializers.py
        r'_urls?\.py$',        # xxx_url.py ou xxx_urls.py
        r'_tasks?\.py$',       # xxx_task.py ou xxx_tasks.py
        r'_migrations?\.py$',  # xxx_migration.py ou xxx_migrations.py
        r'_factory\.py$',      # xxx_factory.py
        r'_cli\.py$',          # xxx_cli.py
        r'_command\.py$',      # xxx_command.py
        r'_manage\.py$',       # xxx_manage.py
        r'_main\.py$',         # xxx_main.py
        r'^__init__\.py$'     # __init__.py
    ]
    for pattern in patterns:
        if re.search(pattern, filename):
            return True
    return False

# Função para identificar arquivos especiais de JavaScript
def is_special_js_file(filename):
    patterns = [
        r'\.schema\.js$',
        r'\.service\.js$',
        r'\.controller\.js$',
        r'\.route\.js$',    # Corrigido de .routes.ts para .route.js
        r'\.routes\.js$',    # Corrigido de .routes.ts para .route.js
        r'\.d\.js$',
        r'\.test\.js$',
        r'\.config\.js$',
        r'\.config\.mjs$',
        r'\.\([^)]+\)\.js$',
        r'\.test\.js$',
        r'\.spec\.js$',
        r'\.config\.js$',
        r'\.setup\.js$',
        r'\.mock\.js$',
        r'\.model\.js$',
        r'\.middleware\.js$',
        r'\.helper\.js$',
        r'\.util\.js$',
        r'\.factory\.js$',
        r'\.validator\.js$',
        r'\.env\.js$',
        r'\.main\.js$',
        r'\.init\.js$',
        r'\.web\.js$'
    ]
    for pattern in patterns:
        if re.search(pattern, filename):
            return True
    return False

# Função para identificar arquivos especiais de qualquer tipo
def is_special_file(filename):
    return (
        is_special_ts_file(filename) or
        is_special_py_file(filename) or
        is_special_js_file(filename)
    )

# Função para listar o diretório recursivamente
def list_directory(path, level=0, output_file=None, exclude_files=None, list_content=False):
    exclude_files = exclude_files or []
    items = sorted(os.listdir(path))

    for item in items:
        full_path = os.path.join(path, item)

        # Ignora arquivos ou pastas que atendem aos critérios
        if should_ignore(item, full_path, exclude_files):
            continue

        if os.path.isdir(full_path):
            output_file = write_to_output(output_file, f"{'  ' * level}|-- {item}/\n")
            output_file = list_directory(full_path, level + 1, output_file, exclude_files, list_content)
        elif os.path.isfile(full_path):
            output_file = write_to_output(output_file, f"{'  ' * level}|-- {item}\n")

            if list_content and (
                item.endswith((
                    '.py', '.js', '.java', '.c', '.cpp', '.h', '.ipynb', '.html', '.css',
                    '.ts', '.tsx', '.scss', '.sass', '.vue', '.dart', '.jsx', '.prisma', '.gql',
                    '.spec.js', '.test.ts', '.test.js', '.github/workflows/.yml', 'Jenkinsfile',
                    'Dockerfile', '.sql', '.db', '.sqlite', '.md', '.rst', '.sh', '.bat', '.ps1', '.cy','.env', '.yaml', '.yml'
                )) or is_special_file(item) or item in ['tsconfig.json', 'package.json', '.env', 'eslint.config.mjs', 'globals.css', '.eslintrc.json', '.prettierrc', '.env.example', 'next.config.ts', 'app.json', 'settings.json', 'requirements.txt', 'netlify.toml', 'globals.css', 'Dockerfile']
            ):

                output_file = write_to_output(output_file, f"{'  ' * (level + 1)}Content:\n")
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            output_file = write_to_output(output_file, f"{'  ' * (level + 2)}{line}")
                except Exception as e:
                    output_file = write_to_output(output_file, f"{'  ' * (level + 2)}[Erro ao ler arquivo: {e}]\n")

    return output_file

# Função para escrever no arquivo de saída, com controle de tamanho
def write_to_output(output_file, content):
    max_file_size = 120000

    if output_file and output_file.tell() + len(content.encode('utf-8')) > max_file_size:
        output_file.close()
        output_file = create_new_output_file(output_file.name)

    output_file.write(content)
    return output_file

# Função para criar um novo arquivo de saída se o limite de tamanho for atingido
def create_new_output_file(file_name):
    base_name, extension = os.path.splitext(file_name)
    n = 1
    new_file_name = f"{base_name}_{n}{extension}"

    while os.path.exists(new_file_name):
        n += 1
        new_file_name = f"{base_name}_{n}{extension}"

    return open(new_file_name, 'w', encoding='utf-8')

# Função para gerar o listing do diretório
def generate_listing(directory_path):
    try:
        current_file_name = os.path.basename(__file__)
    except NameError:
        current_file_name = 'main.py'
    exclude_files = ['directory_listing.txt', 'tree.txt', current_file_name]
    output_file_path = 'directory_listing.txt'
    output_file = open(output_file_path, 'w', encoding='utf-8')

    list_directory(directory_path, output_file=output_file, exclude_files=exclude_files)
    output_file.write("\n\nFile contents:\n\n")
    list_directory(directory_path, output_file=output_file, exclude_files=exclude_files, list_content=True)
    output_file.close()

# Função para gerar a árvore de diretórios
def generate_tree(directory_path):
    try:
        current_file_name = os.path.basename(__file__)
    except NameError:
        current_file_name = 'main.py'
    exclude_files = [
        'directory_listing.txt', 
        'directory_listing_1.txt', 
        'directory_listing_1_1.txt',
        'directory_listing_1_1_1.txt',
        'directory_listing_1_1_1_1.txt',
        'directory_listing_1_1_1_1_1.txt',
        'tree.txt',  
        current_file_name
    ]
    with open('tree.txt', 'w', encoding='utf-8') as tree_file:
        write_tree(directory_path, tree_file, exclude_files)

# Função para escrever a árvore de diretórios
def write_tree(directory_path, tree_file, exclude_files, level=0):
    items = sorted(os.listdir(directory_path))

    for item in items:
        full_path = os.path.join(directory_path, item)

        if should_ignore(item, full_path, exclude_files):
            continue

        tree_file.write(f"{'  ' * level}|-- {item}\n")

        if os.path.isdir(full_path):
            write_tree(full_path, tree_file, exclude_files, level + 1)

# Caminho do diretório a ser listado
directory_path = './'

# Gerando listing e árvore de diretórios
generate_listing(directory_path)
generate_tree(directory_path)

print("The directory listing and tree files have been created.")
