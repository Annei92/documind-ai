def split_into_blocks(text):
    """
    Convert extracted PDF text into smaller logical blocks.

    Blank lines and bullet points are treated as boundaries
    whenever the PDF extraction preserves them.
    """

    lines = text.splitlines()

    blocks = []
    current_block = []

    for line in lines:
        line = line.strip()

        if not line:
            if current_block:
                blocks.append(
                    " ".join(current_block)
                )
                current_block = []

            continue

        
        if (
            line.startswith("•")
            or line.startswith("- ")
            or line.startswith("– ")
        ):
            if current_block:
                blocks.append(
                    " ".join(current_block)
                )
                current_block = []

            blocks.append(line)
            continue

        current_block.append(line)

    if current_block:
        blocks.append(
            " ".join(current_block)
        )

    return blocks


def split_large_block(
    block,
    max_size
):
    """
    Split an oversized block while preferring
    sentence/semicolon boundaries.
    """

    pieces = []

    remaining = block.strip()

    while len(remaining) > max_size:

        candidate = remaining[:max_size]

        # Prefer increasingly weaker boundaries
        boundary = candidate.rfind(". ")

        if boundary == -1:
            boundary = candidate.rfind("; ")

        if boundary == -1:
            boundary = candidate.rfind(", ")

        if boundary == -1:
            boundary = candidate.rfind(" ")

        if boundary == -1:
            boundary = max_size
        else:
            boundary += 1

        piece = remaining[:boundary].strip()

        if piece:
            pieces.append(piece)

        remaining = remaining[
            boundary:
        ].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def chunk_pages(
    pages,
    chunk_size=600,
    overlap_blocks=1
):
    chunks = []

    for page in pages:

        text = page["text"].strip()
        page_number = page["page_number"]

        if not text:
            continue

       

        raw_blocks = split_into_blocks(
            text
        )

        blocks = []

        for block in raw_blocks:

            if len(block) <= chunk_size:
                blocks.append(block)

            else:
                blocks.extend(
                    split_large_block(
                        block,
                        chunk_size
                    )
                )


       

        index = 0

        while index < len(blocks):

            chunk_blocks = []
            chunk_length = 0

            current_index = index

            while current_index < len(blocks):

                block = blocks[
                    current_index
                ]

                additional_length = len(
                    block
                )

                if chunk_blocks:
                    additional_length += 1

                if (
                    chunk_blocks
                    and (
                        chunk_length
                        + additional_length
                        > chunk_size
                    )
                ):
                    break

                chunk_blocks.append(
                    block
                )

                chunk_length += (
                    additional_length
                )

                current_index += 1


            chunk_text = "\n".join(
                chunk_blocks
            ).strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number
                })


            if current_index >= len(blocks):
                break


            

            next_index = max(
                index + 1,
                current_index
                - overlap_blocks
            )

            if next_index <= index:
                next_index = current_index

            index = next_index

    return chunks