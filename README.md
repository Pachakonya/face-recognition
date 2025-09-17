# Face Recognition System

A Python-based face recognition system that uses OpenCV for face detection and PostgreSQL with vector embeddings for face matching and recognition.

## Features

- **Face Detection**: Uses Haar Cascade classifier to detect faces in images
- **Face Extraction**: Automatically crops and saves detected faces
- **Vector Embeddings**: Converts face images to high-dimensional embeddings using imgbeddings
- **Database Storage**: Stores face embeddings in PostgreSQL with vector similarity search
- **Face Matching**: Finds the most similar stored face for a given input image

## Project Structure

```
face_recognition/
├── facereg.py                              # Main face recognition script
├── haarcascade_frontalface_default.xml     # Haar cascade classifier for face detection
├── my_boys.png                            # Source image for face extraction
├── IMG_0913.png                           # Test image for face matching
├── stored-faces/                          # Directory containing extracted face images
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   └── 9.jpg
└── README.md                              # This file
```

## Dependencies

- OpenCV (`cv2`)
- PIL (`Pillow`)
- imgbeddings
- psycopg2
- numpy

## Setup

1. Install required dependencies:
   ```bash
   pip install opencv-python pillow imgbeddings psycopg2-binary numpy
   ```

2. Set up PostgreSQL database with vector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE TABLE pictures (
       picture VARCHAR PRIMARY KEY,
       embedding vector(768)
   );
   ```

3. Set environment variable for database connection:
   ```bash
   export AIVEN_PASSWORD="your_postgresql_connection_string"
   ```

## Usage

The script performs the following operations:

1. **Face Detection & Extraction**:
   - Loads the source image (`my_boys.png`)
   - Detects faces using Haar Cascade classifier
   - Crops and saves each detected face to `stored-faces/` directory

2. **Database Population**:
   - Converts each stored face image to vector embeddings
   - Stores embeddings in PostgreSQL database

3. **Face Matching**:
   - Loads a test image (`IMG_0913.png`)
   - Converts it to vector embedding
   - Finds the most similar face in the database using vector similarity search
   - Displays the matched face image

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in images
2. **Embedding Generation**: Converts face images to 768-dimensional vectors using imgbeddings
3. **Vector Search**: Uses PostgreSQL's vector extension to perform similarity search
4. **Matching**: Finds the closest match using cosine distance between embeddings

## Files Description

- **facereg.py**: Main script containing all face recognition logic
- **haarcascade_frontalface_default.xml**: Pre-trained Haar cascade model for face detection
- **my_boys.png**: Source image containing multiple faces to be extracted
- **IMG_0913.png**: Test image used for face matching
- **stored-faces/**: Contains 10 extracted face images (0.jpg through 9.jpg)

## Notes

- The system currently processes 10 detected faces from the source image
- Face matching uses vector similarity search for accurate results
- Database connection requires proper PostgreSQL setup with vector extension
- Images are displayed using OpenCV's built-in display functions
