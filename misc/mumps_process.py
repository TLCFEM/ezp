from pathlib import Path
from shutil import rmtree


def process_makefile(makefile: Path):
    with open(makefile, "r") as f:
        content = f.read().replace(":", "\n").replace("\\", "\n")
        lines = [x.strip() for x in content.split("\n")]

    items: set[str] = set()
    for line in lines:
        if not line or not line.endswith(".o") or line.count(".") > 1:
            continue
        item_name = line.removesuffix(".o")
        if "$(ARITH)" in item_name:
            for arith in ("s", "d", "c", "z"):
                items.add(item_name.replace("$(ARITH)", arith))
        else:
            items.add(item_name)

    return items


def replace(file: Path):
    return (
        file.read_text()
        .replace("#include <space.h>", '#include "space.h"')
        .replace("# if defined(UPPER) || defined(MUMPS_WIN32)", "#if defined(UPPER)")
        .replace('#include "mumps_int_def.h"', "#define MUMPS_INTSIZE32")
    )


def process(src: str, dest: str):
    dest_folder = Path(dest)
    if dest_folder.exists():
        rmtree(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    src_folder = Path(src)

    for subfolder in ("include", "PORD/include", "PORD/lib"):
        for file in (src_folder / subfolder).glob("*"):
            if file.is_file() and file.name.endswith((".c", ".h")):
                (dest_folder / file.name).write_text(replace(file))

    items = process_makefile(src_folder / "src/Makefile")

    for file in (src_folder / "src").glob("*"):
        if file.is_file() and (name := file.stem) in items:
            file_name = name + file.suffix
            (dest_folder / file_name).write_text(replace(file))

    mumps_c = replace(src_folder / "src/mumps_c.c")
    for arith in ("s", "d", "c", "z"):
        (dest_folder / f"{arith}mumps_c.c").write_text(mumps_c)

    for extra in ("mumps_headers.h", "mumps_save_restore_modes.h", "mumps_tags.h"):
        (dest_folder / extra).write_text(replace(src_folder / "src" / extra))


if __name__ == "__main__":
    pass
