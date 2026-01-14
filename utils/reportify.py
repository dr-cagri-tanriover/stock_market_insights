from typing import Any, Dict, List
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ReportDataType():
    # CAUTION: Do not use underscores in the data type names. They are used to separate the data type from the field index.
    # i.e., use "heading1" instead of "heading_1"
    TABLE="table" # Table data
    IMAGE="image" # Image data
    BODY="body" # Paragraph body text style
    TITLE="title" # Paragraph title text style
    HEADING_1="heading1" # Paragraph heading text style
    HEADING_2="heading2" # Paragraph heading text style

class ReportStyle():
    VSPACE=10   # Vertical space between lines or paragraphes in pdf report
    HSPACE=1   # Horizontal space between report sections

    IMAGE_SCALING_FACTOR=0.90  # 0.95 plot the original image to fit to the pdf page (an additional scaling is done to ensure that!).

class reporter():
    def __init__(self, report_filepath: Path):
        """
            Sample self.report_dict structure:

            {
                <page number>:{
                    "data type"_<field index = 0>: <data>,
                    "data type"_<field index = 1>: <data>,
                    ...
                },
                <page number>:{
                    "data type"_<field index = 0>: <data>,
                    "data type"_<field index = 1>: <data>,
                    ...
                },
                ...
            }

            where <data> can be:
            1 - Table data of the form:
            {
                "header_row": ["col 1 name", "col 2 name", "col 3 name", ...],
                "row 1 str": ["data 1 str value", "data 2 str value", "data 3 str value", ...],
                "row 2 str": ["data 4 str value", "data 5 str value", "data 6 str value", ...],
                ...
            }

            2 - Paragraph data of the form:
                "paragraph text"


            3 - Image data of the form:
                "path to image file"

            You can add more types as needed moving forward.

        """
        class Indices:
            def __init__(self):
                self.page_index = 0
                self.field_index = 0

        self.indices = Indices()
        self.report_filepath = Path(report_filepath)
        self.report_dict = {}  # Report content will be stored in a dictionary. Each page will be a key in this dictionary.
        self.basic_style = getSampleStyleSheet()

    def get_style(self, style_type: ReportDataType):
        
        if style_type == ReportDataType.TITLE:
            return ParagraphStyle(
                    name="TitleCustom",
                    parent=self.basic_style["Title"],
                    fontName="Helvetica-Bold",
                    fontSize=22,
                    leading=26,
                    textColor=colors.darkblue,
                    alignment=TA_CENTER,
                    spaceAfter=18,
            )
        elif style_type == ReportDataType.HEADING_1:
            return ParagraphStyle(
                name="Heading1Custom",
                parent=self.basic_style["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=20,
                textColor=colors.blue,
                alignment=TA_LEFT,
                spaceBefore=12,
                spaceAfter=6,
            )
        elif style_type == ReportDataType.HEADING_2:
            return ParagraphStyle(
                name="Heading2Custom",
                parent=self.basic_style["Heading2"],
                fontName="Helvetica-BoldOblique",  # bold + italic feel
                fontSize=16,
                leading=16,
                textColor=colors.darkmagenta,
                alignment=TA_LEFT,
                spaceBefore=10,
                spaceAfter=4,
            )
        elif style_type == ReportDataType.BODY:
            return ParagraphStyle(
                name="BodyCustom",
                parent=self.basic_style["Normal"],
                fontName="Helvetica",
                fontSize=10.5,
                leading=3,  # line spacing between the lines within a paragraph (useful especially if paragraph spans multiple lines)
                alignment=TA_JUSTIFY,   # full justification
                spaceBefore=0, # space BEFORE the paragraph text
                spaceAfter=0, # space AFTER the paragraph text
                leftIndent=0,
                rightIndent=0,
                firstLineIndent=0,
                textColor=colors.black
            )
        else:
            return getSampleStyleSheet()['Normal']


    def new_page(self, title: str = None):
        """
        Add a new page to the report.
        """

        self.indices.page_index += 1
        self.indices.field_index = 0
        self.report_dict[self.indices.page_index] = {}  # each page index will have a dictionary as value

        if title is not None:
            self.print(ReportDataType.TITLE, title)

    def add_table_data(self, table_dict: Dict[str, List[Any]]):
        """
        Add a table section to the report.

        Args:
            table_dict: Dictionary containing the table data
        """
        
        self.indices.field_index += 1  # Data will be added to a new field.
        new_key = f"{ReportDataType.TABLE}_{self.indices.field_index}"
        self.report_dict[self.indices.page_index][new_key] = table_dict


    """
    def add_paragraph_data(self, paragraph_str: str):
        # Add a paragraph section to the report.

        self.indices.field_index += 1  # Data will be added to a new field.
        new_key = f"{ReportDataType.PARAGRAPH}_{self.indices.field_index}"
        self.report_dict[self.indices.page_index][new_key] = paragraph_str
    """
    
    def print_image(self, image_filepath: Path):
        """
        Add an image section to the report. Each string value in image_dict will be a path to an image file.
        """

        self.indices.field_index += 1  # Data will be added to a new field.
        new_key = f"{ReportDataType.IMAGE}_{self.indices.field_index}"
        self.report_dict[self.indices.page_index][new_key] = image_filepath

    def print(self, data_type: ReportDataType, string_data: str):
        """
        Print the user requested text type to the console as well as the pdf report
        """
        
        print(string_data)

        self.indices.field_index += 1  # Data will be added to a new field.
        new_key = f"{data_type}_{self.indices.field_index}"
        self.report_dict[self.indices.page_index][new_key] = string_data

    def generate_report(self):

        """
        Generate the report.
        """

        # Create the report directory if it doesn't exist
        self.report_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create the report file
        doc = SimpleDocTemplate(str(self.report_filepath), pagesize=letter) 

        content_list = []       # Report content will be parsed and appended to this list 

        # Add the report content to the document
        for page_index, page_data in self.report_dict.items():
            
            for data_type_key, data in page_data.items():

                data_type = data_type_key.split("_")[0]  # separate the data type string from the field index
                if data_type == ReportDataType.TABLE:
                    # Write table type data to report
                    #TBD
                    table = Table(data)
                    content_list.append(table)
                elif data_type == ReportDataType.IMAGE:
                    # Write image type data to report
                    image = Image(str(data))
                    
                    # Scale the image to the desired width and height before dumping into the pdf report
                    page_scale = 1.0  # Assuming the original image already fits the pdf page
                    if (image.imageHeight > image.imageWidth):
                        # For portrait images, scale the image height to the supported page height
                        if (image.imageHeight > doc.height):
                            # Oops! image height does not fit the page! Need to scale.
                            page_scale = doc.height / image.imageHeight  # scaling factor updated to fit the original image height to page.
                    else:
                        # For landscape images, scale the image width to the supported page width
                        if (image.imageWidth > doc.width):
                            # Oops! image width does not fit the page! Need to scale.
                            page_scale = doc.width / image.imageWidth  # scaling factor updated to fit the original image width to page.

                    # Scale by the user requested scaling factor as well!
                    image.drawWidth = image.imageWidth * ReportStyle.IMAGE_SCALING_FACTOR * page_scale
                    image.drawHeight = image.imageHeight * ReportStyle.IMAGE_SCALING_FACTOR * page_scale

                    content_list.append(image)
                    content_list.append(Spacer(ReportStyle.HSPACE, ReportStyle.VSPACE))
                else:
                    # All other text style data is handled by the get_style method
                    # Write paragraph type data to report
                    paragraph = Paragraph(data, self.get_style(data_type))
                    content_list.append(paragraph)
                    content_list.append(Spacer(ReportStyle.HSPACE, ReportStyle.VSPACE))

            if page_index < len(self.report_dict):
                # Add a page break before each page except the last one
                content_list.append(PageBreak())

        # Add the complete report content to the document
        doc.build(content_list)

        pass

