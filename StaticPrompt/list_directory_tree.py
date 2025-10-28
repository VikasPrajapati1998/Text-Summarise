import os

def list_directory_tree(start_path, indent=0):
    """Recursively prints the directory tree structure."""
    try:
        # Get all entries in the directory
        entries = os.listdir(start_path)
    except PermissionError:
        print(" " * indent + f"[Access Denied]: {start_path}")
        return

    for entry in entries:
        path = os.path.join(start_path, entry)
        if os.path.isdir(path):
            print(" " * indent + f"[DIR]  {entry}")
            list_directory_tree(path, indent + 4)
        else:
            print(" " * indent + f"- {entry}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path to scan: ").strip()

    if not os.path.exists(folder_path):
        print("❌ Error: The provided path does not exist.")
    else:
        print(f"\n📂 Directory structure of: {folder_path}\n")
        list_directory_tree(folder_path)
