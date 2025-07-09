#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to remove emoticons from a text file.
This script accepts a file path as input, removes emoticon characters,
and saves the cleaned text back to the file.
"""

import re
import sys
import argparse
import unicodedata

def is_emoticon(char):
    """
    Check if a character is an emoticon/emoji.
    
    Args:
        char (str): A single character to check
        
    Returns:
        bool: True if the character is an emoticon, False otherwise
    """
    if len(char) == 0:
        return False
        
    # Check if character is in emoji blocks
    category = unicodedata.category(char)
    
    # Check for emoji blocks based on unicode properties
    # So - Symbol, Other
    # Sk - Symbol, Modifier
    if category == 'So' or category == 'Sk':
        return True
        
    # Check for specific emoji ranges
    code = ord(char)
    # Basic emoticons
    if 0x1F600 <= code <= 0x1F64F:  # Emoticons block
        return True
    elif 0x1F300 <= code <= 0x1F5FF:  # Miscellaneous Symbols and Pictographs
        return True
    elif 0x1F680 <= code <= 0x1F6FF:  # Transport and Map Symbols
        return True
    elif 0x2600 <= code <= 0x26FF:  # Miscellaneous Symbols
        return True
    elif 0x2700 <= code <= 0x27BF:  # Dingbats
        return True
    elif 0x1F900 <= code <= 0x1F9FF:  # Supplemental Symbols and Pictographs
        return True
    elif 0x1F1E6 <= code <= 0x1F1FF:  # Regional Indicator Symbols (flags)
        return True
    
    return False

def remove_emoticons(input_text):
    """
    Remove emoticons from text.
    
    Args:
        input_text (str): Text containing emoticons
        
    Returns:
        str: Text with emoticons removed
    """
    result = ""
    for char in input_text:
        if not is_emoticon(char):
            result += char
    return result

def process_file(file_path):
    """
    Process a file to remove emoticons.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Remove emoticons
        cleaned_content = remove_emoticons(content)
        
        # Write back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
            
        return True, f"Successfully processed {file_path}"
    except UnicodeDecodeError:
        # Try with a different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                
            cleaned_content = remove_emoticons(content)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
                
            return True, f"Successfully processed {file_path} (with latin-1 encoding)"
        except Exception as e:
            return False, f"Error processing file with latin-1 encoding: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Remove emoticons from a text file")
    parser.add_argument("file_path", help="Path to the file to process")
    args = parser.parse_args()
    
    # Process the file
    success, message = process_file(args.file_path)
    
    # Output the result
    if success:
        print(message)
    else:
        print(message, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()