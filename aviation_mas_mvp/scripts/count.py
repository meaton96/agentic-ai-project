import os
from pathlib import Path

def count_python_loc(directory):
    directory_path = Path(directory)
    
    # Ensure the directory exists
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return

    total_files = 0
    total_lines = 0
    blank_lines = 0
    comment_lines = 0
    code_lines = 0

    # rglob("*.py") recursively searches for all Python files
    for py_file in directory_path.rglob("*.py"):
        total_files += 1
        try:
            # Explicitly use utf-8 to avoid decoding errors on different OSs
            with open(py_file, 'r', encoding='utf-8') as f:
                for line in f:
                    total_lines += 1
                    stripped_line = line.strip()
                    
                    if not stripped_line:
                        blank_lines += 1
                    elif stripped_line.startswith('#'):
                        comment_lines += 1
                    else:
                        code_lines += 1
        except Exception as e:
            print(f"Warning: Could not read {py_file} - {e}")

    # Print the results
    print(f"\n--- LoC Summary for '{directory}' ---")
    print(f"Python files scanned: {total_files}")
    print(f"Total lines:          {total_lines}")
    print(f"Blank lines:          {blank_lines}")
    print(f"Comment lines:        {comment_lines}")
    print(f"Actual code lines:    {code_lines}")
    print("-" * 37)

if __name__ == "__main__":
    # You can change '.' to any hardcoded path, or leave it to prompt the user
    target_dir = input("Enter the directory path to scan (or press Enter for current directory): ")
    
    if not target_dir.strip():
        target_dir = "."
        
    count_python_loc(target_dir)