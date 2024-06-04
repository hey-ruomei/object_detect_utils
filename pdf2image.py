import os
import fitz
import shutil

def pdf2img_main(pdf_file, image_path):
    i = 21
    pdfDoc = fitz.open(pdf_file)
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        rotate = int(0)
        zoom_x = 2.0
        zoom_y = 2.0
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(os.path.join(image_path, str(i) + ".jpg"))
        i += 1
if (__name__ == '__main__'):
    pdf_file = "./test.pdf"
    image_path = "./test_watermark"
    pdf2img_main(pdf_file, image_path)