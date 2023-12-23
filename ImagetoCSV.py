from PIL import Image
import csv
import os

def image_to_csv(image_path, csv_path):
    # Open the image file
    img = Image.open(image_path)
    img = img.convert('RGB')  # Convert image to RGB if it's in another mode
    
    # Get image dimensions
    width, height = img.size
    
    # Create a CSV file for storing pixel data
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header with column names (optional)
        csv_writer.writerow(['R', 'G', 'B'])
        
        # Iterate through each pixel and write RGB values to CSV
        for y in range(height):
            for x in range(width):
                r, g, b = img.getpixel((x, y))
                csv_writer.writerow([r, g, b])

# Path to the folder containing images
relpath = 'Catla/Body'
folder_path = 'Datasets/Fish/' + relpath

# Path to the folder where CSV files will be saved
csv_folder_path = 'Datasets/FishCSV/' + relpath

# Create the CSV folder if it doesn't exist
if not os.path.exists(csv_folder_path):
    os.makedirs(csv_folder_path)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        print('Converting', filename)
        image_path = os.path.join(folder_path, filename)
        csv_path = os.path.join(csv_folder_path, os.path.splitext(filename)[0] + '.csv')
        image_to_csv(image_path, csv_path)