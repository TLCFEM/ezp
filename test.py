import os


def run(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.startswith("d") and f"s{file[1:]}" in files:
                if os.path.exists(
                    f"/home/theodore/Downloads/MUMPS_5.7.3/src/z{file[1:]}"
                ):
                    print(f"z{file[1:]}")
                if os.path.exists(
                    f"/home/theodore/Downloads/MUMPS_5.7.3/src/c{file[1:]}"
                ):
                    print(f"c{file[1:]}")


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__)) + "/mumps"
    run(folder)
