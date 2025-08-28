# Schedule-time-table
Python Libraries: numpy, pandas, pytesseract, ics, scikit-learn

The objective of this project is to develop an Automatic Web Event Scheduler that can autonomously extract and organize event information from images and schedule them using digital calendar services. The system begins by preprocessing input images—typically containing tabular schedules such as class timetables or event agendas—to enhance clarity for optical character recognition. 

Using Tesseract OCR, the text is extracted from the image while preserving spatial layout. To reconstruct the table structure, hierarchical clustering is applied to group related text segments into coherent columns and rows. Once the structure is identified, the system parses essential information such as event titles, days, and time slots using pattern recognition techniques. This structured data is then formatted and integrated with the Google Calendar API to automatically create calendar events, thereby eliminating the need for manual scheduling. 

To enhance performance, the project incorporates machine learning models aimed at improving OCR accuracy, refining clustering outcomes, and adapting to diverse table formats. Overall, this project combines elements of computer vision, natural language processing, and automation to deliver a robust and efficient solution for converting visual event data into actionable calendar entries.
