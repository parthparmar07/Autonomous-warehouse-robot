import hashlib
import platform
import subprocess
from pathlib import Path

def provenance(root):
    root=Path(root)
    files=sorted(list((root/'src').rglob('*.py'))+list((root/'scripts').rglob('*.py'))+list((root/'configs').rglob('*.yaml'))+list(root.glob('requirements*.txt')))
    digest=hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).replace('\\','/').encode());digest.update(path.read_bytes())
    def git(*args):
        return subprocess.run(['git',*args],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    try:revision,dirty=git('rev-parse','HEAD'),bool(git('status','--porcelain'))
    except (OSError,subprocess.CalledProcessError):revision,dirty=None,None
    return dict(source_sha256=digest.hexdigest(),git_revision=revision,working_tree_dirty=dirty,
        python=platform.python_version(),platform=platform.platform(),processor=platform.processor())
