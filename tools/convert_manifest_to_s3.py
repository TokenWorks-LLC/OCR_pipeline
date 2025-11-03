"""
Convert manifest from Google Drive paths to S3 paths.
Reads the original manifest and replaces G:\ paths with s3:// paths.
"""
import sys
import pathlib

def convert_path_to_s3(local_path: str, s3_bucket: str, s3_prefix: str) -> str:
    """Convert local/Google Drive path to S3 path."""
    # Extract just the filename part after "Secondary Sources"
    # data\drive\Secondary Sources\file.pdf -> file.pdf
    # G:\.shortcut-targets-by-id\...\Secondary Sources\file.pdf -> file.pdf
    path_normalized = local_path.replace("\\", "/")
    parts = path_normalized.split("/Secondary Sources/")
    if len(parts) < 2:
        # If path doesn't contain "Secondary Sources", just use the basename
        filename = pathlib.Path(local_path).name
    else:
        filename = parts[1]
    
    # Construct S3 path
    s3_path = f"s3://{s3_bucket}/{s3_prefix}{filename}"
    return s3_path


def main():
    if len(sys.argv) < 4:
        print("Usage: python convert_manifest_to_s3.py <input_manifest> <output_manifest> <s3_bucket> [s3_prefix]")
        print("Example: python convert_manifest_to_s3.py old.txt new.txt ocr-page-text-005466605994 pdfs/secondary-sources/")
        sys.exit(1)
    
    input_manifest = pathlib.Path(sys.argv[1])
    output_manifest = pathlib.Path(sys.argv[2])
    s3_bucket = sys.argv[3]
    s3_prefix = sys.argv[4] if len(sys.argv) > 4 else "pdfs/secondary-sources/"
    
    # Ensure prefix ends with /
    if s3_prefix and not s3_prefix.endswith("/"):
        s3_prefix += "/"
    
    print(f"Converting manifest: {input_manifest}")
    print(f"S3 bucket: {s3_bucket}")
    print(f"S3 prefix: {s3_prefix}")
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_manifest, 'r', encoding='utf-8') as infile, \
         open(output_manifest, 'w', encoding='utf-8') as outfile:
        
        # Read and write header
        header = infile.readline()
        outfile.write(header)
        
        # Process each line
        for line_num, line in enumerate(infile, start=2):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"Warning: Line {line_num} has incorrect format: {line}")
                skipped_count += 1
                continue
            
            pdf_path = parts[0]
            page_no = parts[1]
            
            # Convert path if it's a local or Google Drive path
            if pdf_path.startswith("G:\\") or pdf_path.startswith("data\\") or pdf_path.startswith("data/"):
                s3_path = convert_path_to_s3(pdf_path, s3_bucket, s3_prefix)
                outfile.write(f"{s3_path}\t{page_no}\n")
                converted_count += 1
            else:
                # Keep as-is if already an S3 path or other format
                outfile.write(f"{pdf_path}\t{page_no}\n")
                skipped_count += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {converted_count} entries")
    print(f"   Skipped: {skipped_count} entries")
    print(f"   Output: {output_manifest}")


if __name__ == "__main__":
    main()
