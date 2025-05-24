#!/usr/bin/env python3
"""
Storage Cleanup Script for Big Data Migrator

This script helps clean up accumulated files that may be taking up disk space:
- Old conversation files
- Large log files
- Temporary files
- Cached data
"""
import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

def main():
    print("🧹 Big Data Migrator - Storage Cleanup")
    print("="*50)
    print()
    
    # Get current storage usage
    total_size = calculate_directory_size(".")
    print(f"📊 Current project size: {format_size(total_size)}")
    print()
    
    # Analyze storage usage by directory
    print("📁 Storage usage by directory:")
    directories = ["conversations", "logs", "temp", "uploads", "exports"]
    
    for directory in directories:
        if os.path.exists(directory):
            dir_size = calculate_directory_size(directory)
            file_count = count_files(directory)
            print(f"  {directory}: {format_size(dir_size)} ({file_count} files)")
        else:
            print(f"  {directory}: Not found")
    print()
    
    # Ask user what to clean
    print("🎯 What would you like to clean?")
    print("1. Old conversations (older than 7 days)")
    print("2. Large log files (keep last 100KB)")
    print("3. Temporary files")
    print("4. All of the above")
    print("5. Custom cleanup")
    print("6. Exit")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        cleanup_old_conversations()
    elif choice == "2":
        cleanup_large_logs()
    elif choice == "3":
        cleanup_temp_files()
    elif choice == "4":
        cleanup_old_conversations()
        cleanup_large_logs()
        cleanup_temp_files()
    elif choice == "5":
        custom_cleanup()
    else:
        print("👋 Exiting cleanup")
        return
    
    # Show results
    new_total_size = calculate_directory_size(".")
    freed_space = total_size - new_total_size
    print()
    print(f"✅ Cleanup complete!")
    print(f"📉 Freed space: {format_size(freed_space)}")
    print(f"📊 New project size: {format_size(new_total_size)}")

def cleanup_old_conversations():
    """Clean up conversation files older than 7 days."""
    print("🗂️ Cleaning old conversations...")
    
    if not os.path.exists("conversations"):
        print("  No conversations directory found")
        return
    
    cutoff_date = datetime.now() - timedelta(days=7)
    removed_count = 0
    freed_space = 0
    
    for filename in os.listdir("conversations"):
        if filename.endswith(".json"):
            filepath = os.path.join("conversations", filename)
            
            # Check file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            if file_time < cutoff_date:
                file_size = os.path.getsize(filepath)
                os.remove(filepath)
                removed_count += 1
                freed_space += file_size
                print(f"    Removed: {filename}")
    
    print(f"  📊 Removed {removed_count} old conversations")
    print(f"  💾 Freed: {format_size(freed_space)}")

def cleanup_large_logs():
    """Clean up large log files, keeping only recent entries."""
    print("📋 Cleaning large log files...")
    
    if not os.path.exists("logs"):
        print("  No logs directory found")
        return
    
    freed_space = 0
    
    for filename in os.listdir("logs"):
        if filename.endswith(".log"):
            filepath = os.path.join("logs", filename)
            file_size = os.path.getsize(filepath)
            
            # If file is larger than 100KB, truncate it
            if file_size > 100 * 1024:  # 100KB
                print(f"    Truncating large log: {filename} ({format_size(file_size)})")
                
                # Keep only last 100KB of the file
                with open(filepath, 'rb') as f:
                    f.seek(-100 * 1024, 2)  # Seek to 100KB from end
                    content = f.read()
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                new_size = os.path.getsize(filepath)
                freed_space += (file_size - new_size)
    
    print(f"  💾 Freed: {format_size(freed_space)}")

def cleanup_temp_files():
    """Clean up temporary files."""
    print("🗃️ Cleaning temporary files...")
    
    directories_to_clean = ["temp", "tmp"]
    freed_space = 0
    removed_count = 0
    
    for directory in directories_to_clean:
        if not os.path.exists(directory):
            continue
            
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Skip if it's a directory
            if os.path.isdir(filepath):
                continue
            
            # Remove temporary files
            file_size = os.path.getsize(filepath)
            os.remove(filepath)
            removed_count += 1
            freed_space += file_size
            print(f"    Removed: {filename}")
    
    print(f"  📊 Removed {removed_count} temporary files")
    print(f"  💾 Freed: {format_size(freed_space)}")

def custom_cleanup():
    """Allow user to specify custom cleanup options."""
    print("⚙️ Custom cleanup options:")
    print()
    
    # Show conversation statistics
    if os.path.exists("conversations"):
        conv_files = [f for f in os.listdir("conversations") if f.endswith(".json")]
        print(f"📁 Conversations: {len(conv_files)} files")
        
        if len(conv_files) > 10:
            keep_count = input(f"  Keep how many recent conversations? (default: 10): ").strip()
            try:
                keep_count = int(keep_count) if keep_count else 10
                cleanup_conversations_keep_recent(keep_count)
            except ValueError:
                print("  Invalid number, skipping conversation cleanup")
    
    # Log file size limit
    if os.path.exists("logs"):
        log_limit = input("  Maximum log file size in KB? (default: 100): ").strip()
        try:
            log_limit = int(log_limit) if log_limit else 100
            cleanup_logs_with_limit(log_limit * 1024)  # Convert to bytes
        except ValueError:
            print("  Invalid number, skipping log cleanup")

def cleanup_conversations_keep_recent(keep_count: int):
    """Keep only the most recent N conversations."""
    if not os.path.exists("conversations"):
        return
    
    conv_files = []
    for filename in os.listdir("conversations"):
        if filename.endswith(".json"):
            filepath = os.path.join("conversations", filename)
            mtime = os.path.getmtime(filepath)
            conv_files.append((mtime, filepath, filename))
    
    # Sort by modification time (newest first)
    conv_files.sort(reverse=True)
    
    # Remove old files
    freed_space = 0
    removed_count = 0
    
    for i, (mtime, filepath, filename) in enumerate(conv_files):
        if i >= keep_count:  # Keep only first keep_count files
            file_size = os.path.getsize(filepath)
            os.remove(filepath)
            removed_count += 1
            freed_space += file_size
            print(f"    Removed old conversation: {filename}")
    
    print(f"  📊 Kept {min(len(conv_files), keep_count)} recent conversations")
    print(f"  📊 Removed {removed_count} old conversations")
    print(f"  💾 Freed: {format_size(freed_space)}")

def cleanup_logs_with_limit(max_size_bytes: int):
    """Clean up logs with specified size limit."""
    if not os.path.exists("logs"):
        return
    
    freed_space = 0
    
    for filename in os.listdir("logs"):
        if filename.endswith(".log"):
            filepath = os.path.join("logs", filename)
            file_size = os.path.getsize(filepath)
            
            if file_size > max_size_bytes:
                print(f"    Truncating: {filename} ({format_size(file_size)})")
                
                # Keep only last max_size_bytes of the file
                with open(filepath, 'rb') as f:
                    f.seek(-max_size_bytes, 2)  # Seek to max_size_bytes from end
                    content = f.read()
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                new_size = os.path.getsize(filepath)
                freed_space += (file_size - new_size)
    
    print(f"  💾 Freed: {format_size(freed_space)}")

def calculate_directory_size(directory: str) -> int:
    """Calculate total size of a directory in bytes."""
    if not os.path.exists(directory):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass  # Skip files we can't access
    
    return total_size

def count_files(directory: str) -> int:
    """Count number of files in a directory."""
    if not os.path.exists(directory):
        return 0
    
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
    
    return file_count

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} TB"

if __name__ == "__main__":
    main() 