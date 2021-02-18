from openpyxl.styles import PatternFill, Border, Side, Alignment, Protection, Font

HEADER_FONT = Font(bold=True)
HEADER_FILL = PatternFill(start_color="00C0C0C0", end_color="00C0C0C0", fill_type="solid")
HEADER_BORDER = Border(left=Side(style='thick'), right=Side(style='thick'), top=Side(style='thick'),
                       bottom=Side(style='thick'))
