import os
import glob
import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. DATASET FOLDER
# ============================================================

DATASET_FOLDER = "dataset"


# ============================================================
# 2. FIND PDF AND JSON FILES
# ============================================================

pdf_files = glob.glob(os.path.join(DATASET_FOLDER, "**", "*.pdf"), recursive=True)
json_files = glob.glob(os.path.join(DATASET_FOLDER, "**", "*.json"), recursive=True)

print("======================================")
print("LEGAL DATASET")
print("======================================")

print("\nPDF files found:", len(pdf_files))

for pdf in pdf_files:
    print(" -", pdf)

print("\nJSON files found:", len(json_files))

for json_file in json_files:
    print(" -", json_file)


# ============================================================
# 3. CHUNK SETTINGS
# ============================================================

chunk_size = 500
overlap = 100

all_chunks = []
all_metadata = []


# ============================================================
# 4. PROCESS PDF FILES
# ============================================================

print("\n======================================")
print("PROCESSING PDF FILES")
print("======================================")

for pdf_file in pdf_files:

    pdf_name = os.path.basename(pdf_file)

    print("\nReading PDF:", pdf_name)

    try:
        reader = PdfReader(pdf_file)

        pdf_text = ""

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()

                if text:
                    pdf_text += text + "\n"

            except Exception as e:
                print(f"Could not read page {page_number}: {e}")

        words = pdf_text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:

                all_chunks.append(
                    f"Source: {pdf_name}\n\n{chunk_text}"
                )

                all_metadata.append({
                    "source": pdf_name,
                    "type": "PDF"
                })

            start = end - overlap

        print("Chunks created from PDF:", pdf_name)

    except Exception as e:
        print("Error processing PDF:", pdf_name)
        print(e)


# ============================================================
# 5. PROCESS JSON FILES
# ============================================================

print("\n======================================")
print("PROCESSING JSON FILES")
print("======================================")

for json_file in json_files:

    json_name = os.path.basename(json_file)

    print("\nReading JSON:", json_name)

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ----------------------------------------------------
        # JSON can be a list OR a dictionary
        # ----------------------------------------------------

        if isinstance(data, dict):

            # If the actual records are stored inside a key
            # such as "data", "questions", "records", etc.
            possible_keys = [
                "data",
                "questions",
                "records",
                "items"
            ]

            records = None

            for key in possible_keys:

                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break

            # If no list was found, treat dictionary as one record
            if records is None:
                records = [data]

        elif isinstance(data, list):

            records = data

        else:

            print("Unsupported JSON format:", json_name)
            continue


        json_count = 0

        # ----------------------------------------------------
        # Process each question-answer record
        # ----------------------------------------------------

        for record in records:

            if not isinstance(record, dict):
                continue

            question = record.get("question", "")
            answer = record.get("answer", "")

            # Convert values to strings
            question = str(question).strip()
            answer = str(answer).strip()

            # Skip empty records
            if not question and not answer:
                continue

            # ------------------------------------------------
            # Create searchable legal text
            # ------------------------------------------------

            if question and answer:

                legal_text = (
                    f"Question: {question}\n\n"
                    f"Answer: {answer}"
                )

            elif question:

                legal_text = f"Question: {question}"

            else:

                legal_text = f"Answer: {answer}"


            # ------------------------------------------------
            # Chunk JSON answer if it is very large
            # ------------------------------------------------

            words = legal_text.split()

            start = 0

            while start < len(words):

                end = start + chunk_size

                chunk_text = " ".join(words[start:end]).strip()

                if chunk_text:

                    all_chunks.append(
                        f"Source: {json_name}\n\n{chunk_text}"
                    )

                    all_metadata.append({
                        "source": json_name,
                        "type": "JSON",
                        "question": question
                    })

                start = end - overlap

            json_count += 1

        print("JSON records processed:", json_count)


    except json.JSONDecodeError as e:

        print("Invalid JSON file:", json_name)
        print(e)

    except Exception as e:

        print("Error processing JSON:", json_name)
        print(e)


# ============================================================
# 6. CHECK TOTAL CHUNKS
# ============================================================

print("\n======================================")
print("CHUNKING COMPLETED")
print("======================================")

print("Total chunks created:", len(all_chunks))

if len(all_chunks) == 0:

    print("\nERROR: No chunks were created.")
    print("Check your dataset folder and files.")
    exit()


# ============================================================
# 7. SAVE CHUNKS
# ============================================================

print("\nSaving chunks...")

with open(
    "chunk_texts.txt",
    "w",
    encoding="utf-8"
) as f:

    for i, chunk in enumerate(all_chunks):

        f.write(f"CHUNK ID: {i}\n")
        f.write(chunk)
        f.write("\n\n")
        f.write("=" * 80)
        f.write("\n\n")


print("chunk_texts.txt created successfully.")


# ============================================================
# 8. SAVE METADATA
# ============================================================

print("Saving metadata...")

with open(
    "chunk_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_metadata,
        f,
        indent=4,
        ensure_ascii=False
    )


print("chunk_metadata.json created successfully.")


# ============================================================
# 9. LOAD EMBEDDING MODEL
# ============================================================

print("\n======================================")
print("LOADING EMBEDDING MODEL")
print("======================================")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 10. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings in batches...")

BATCH_SIZE = 32
CHECKPOINT_FILE = "embeddings_checkpoint.npy"

all_embeddings = []

total_chunks = len(all_chunks)

for start in range(0, total_chunks, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total_chunks)

    print(f"\nEmbedding chunks {start} to {end} of {total_chunks}")

    batch_embeddings = model.encode(
        all_chunks[start:end],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    batch_embeddings = np.asarray(
        batch_embeddings,
        dtype="float32"
    )

    all_embeddings.append(batch_embeddings)

    # Save checkpoint after every batch
    embeddings_so_far = np.vstack(all_embeddings)

    np.save(
        CHECKPOINT_FILE,
        embeddings_so_far
    )

    print(
        f"Checkpoint saved: {len(embeddings_so_far)} embeddings"
    )


embeddings = np.vstack(all_embeddings)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 11. CREATE FAISS INDEX
# ============================================================

print("\n======================================")
print("CREATING FAISS INDEX")
print("======================================")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# ============================================================
# 12. SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    "legal_vectors.index"
)

print("\nFAISS index saved as:")
print("legal_vectors.index")


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n======================================")
print("PROCESS COMPLETED SUCCESSFULLY")
print("======================================")

print("PDF files processed   :", len(pdf_files))
print("JSON files processed  :", len(json_files))
print("Total chunks created  :", len(all_chunks))
print("Vectors stored        :", index.ntotal)

print("\nFiles created:")

print("1. chunk_texts.txt")
print("2. chunk_metadata.json")
print("3. legal_vectors.index")

print("\nYour legal dataset is now ready for semantic search.")