from openpyxl.styles import PatternFill, Border, Side, Alignment, Protection, Font

HEADER_FONT = Font(bold=True)
HEADER_FILL = PatternFill(start_color="AA00CC", end_color="AA00CC", fill_type="solid")
HEADER_BORDER = Border(left=Side(style='thick'), right=Side(style='thick'), top=Side(style='thick'),
                       bottom=Side(style='thick'))
