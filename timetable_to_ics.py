import pandas as pd
from ics import Calendar, Event
from datetime import datetime, timedelta
import re
import pytz

def parse_time_range(time_str: str):
    """
    Given a string like '10-11AM' or '11-12PM', returns ('10AM','11AM') or ('11AM','12PM'), etc.
    Raises ValueError if it doesn’t match the expected pattern.
    """
    m = re.match(r'^\s*(\d{1,2})-(\d{1,2})(AM|PM)\s*$', time_str.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid time-range format: {time_str!r}")

    h1, h2, end_suffix = m.group(1), m.group(2), m.group(3).upper()
    h1_i, h2_i = int(h1), int(h2)

    # Decide start-suffix: if the ending suffix is PM at 12 → start was AM (crossing noon);
    # if ending suffix is AM at 12 → start was PM (crossing midnight);
    # otherwise it’s the same suffix as the end.
    if end_suffix == 'PM' and h2_i == 12:
        start_suffix = 'AM'
    elif end_suffix == 'AM' and h2_i == 12:
        start_suffix = 'PM'
    else:
        start_suffix = end_suffix

    start_time = f"{h1_i}{start_suffix}"
    end_time   = f"{h2_i}{end_suffix}"
    return start_time, end_time


def parse_time(time_str):
    """Convert a time string like '10AM' or '2PM' to hours and minutes."""
    # Extract hours, minutes, and period (AM/PM)
    match = re.search(r'(\d+)(?::(\d+))?([AP]M)', time_str, re.IGNORECASE)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        period = match.group(3).upper()
        
        # Convert to 24-hour format
        if period == 'PM' and hours < 12:
            hours += 12
        elif period == 'AM' and hours == 12:
            hours = 0
            
        return hours, minutes
    
    return None, None


def get_next_weekday(d, weekday):
    """Get the next occurrence of a weekday from date d."""
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return d + timedelta(days_ahead)


def day_abbr_to_weekday(day_abbr):
    """Convert day abbreviation to weekday number (0=Monday, 6=Sunday)."""
    day_map = {
        'mon': 0, 'monday': 0,
        'tue': 1, 'tues': 1, 'tuesday': 1,
        'wed': 2, 'weds': 2, 'wednesday': 2,
        'thu': 3, 'thur': 3, 'thurs': 3, 'thus': 3, 'thursday': 3,
        'fri': 4, 'friday': 4,
        'sat': 5, 'saturday': 5,
        'sun': 6, 'sunday': 6
    }
    return day_map.get(day_abbr.lower(), None)

def timetable_to_ics(df, timezone='UTC', output_file='timetable.ics'):
    """
    Convert a timetable DataFrame to an ICS file
    
    Parameters:
    df (DataFrame): Pandas DataFrame with days in first column and time slots as column names
    timezone (str): Timezone for the events (default: 'UTC')
    output_file (str): Name of the output ICS file
    """
    cal = Calendar()
    tz = pytz.timezone(timezone)
    
    # Get the next Monday as a reference date
    today = datetime.now()
    next_monday = today
    while next_monday.weekday() != 0:  # 0 is Monday
        next_monday = next_monday + timedelta(days=1)
    
    # Figure out the name of the day column
    day_column = None
    for col in df.columns:
        if col.lower() in ['day', 'days', 'day/time', 'weekday']:
            day_column = col
            break
    
    if day_column is None and df.columns[0].lower() not in ['day', 'days', 'day/time', 'weekday']:
        # If no day column found, assume the first column is the day column
        day_column = df.columns[0]
    
    # For each row in the DataFrame
    for idx, row in df.iterrows():
        day_value = row[day_column]
        if isinstance(day_value, str):
            day = day_value
        else:
            # If not a string, try to get a string representation
            day = str(day_value)
        
        weekday = day_abbr_to_weekday(day)
        
        if weekday is None:
            print(f"Warning: Could not parse day '{day}'. Skipping.")
            continue
        
        # Calculate the date for this weekday
        event_date = get_next_weekday(next_monday, weekday)
        
        # Process each time slot
        for time_slot, event_name in row.items():
            # Skip the day column and empty cells
            if time_slot == day_column or pd.isna(event_name) or str(event_name).strip() == 'None':
                continue
            
            # Parse the course name - extract from patterns like "NLP{IT-603)"
            if isinstance(event_name, str):
                # Clean up the event name (handle special characters and braces)
                event_name = re.sub(r'[{}]', '(', event_name)  # Replace { with (
                event_name = re.sub(r'[}]', ')', event_name)   # Replace } with )
                event_name = re.sub(r'[—]', '-', event_name)   # Replace em dash with hyphen
                
                # Clean up the time slot string if needed
                time_slot = str(time_slot).strip()
            
            # Parse the time slot
            start_time_str, end_time_str = parse_time_range(time_slot)
            if not start_time_str or not end_time_str:
                print(f"Warning: Could not parse time slot '{time_slot}'. Skipping.")
                continue
            
            # Get hours and minutes
            start_hours, start_minutes = parse_time(start_time_str)
            end_hours, end_minutes = parse_time(end_time_str)
            
            if start_hours is None or end_hours is None:
                print(f"Warning: Could not parse times in '{time_slot}'. Skipping.")
                continue
            
            # Fix the end time if it appears to be before the start time
            if end_hours < start_hours:
                end_hours += 12  # Assume PM if the end time seems earlier than start time
            
            # Create datetime objects for start and end
            start_datetime = tz.localize(datetime(
                event_date.year, 
                event_date.month, 
                event_date.day, 
                start_hours, 
                start_minutes
            ))
            
            end_datetime = tz.localize(datetime(
                event_date.year, 
                event_date.month, 
                event_date.day, 
                end_hours, 
                end_minutes
            ))
            
            # Ensure end is after begin
            if end_datetime <= start_datetime:
                print(f"Warning: End time must be after start time for '{event_name}' on {day} at {time_slot}. Adjusting...")
                end_datetime = start_datetime + timedelta(hours=1)  # Default to 1 hour if times are invalid
            
            # Create an event
            e = Event()
            e.name = event_name
            e.begin = start_datetime
            e.end = end_datetime
            
            # Add the event to the calendar
            cal.events.add(e)
    
    # Write the calendar to a file
    with open(output_file, 'w') as f:
        f.write(str(cal))
    
    return output_file

# Example usage:
# Assuming your DataFrame looks like:
# 
#   days    10-11AM    11AM-12PM    1-2PM
# --------------------------------------
#   Mon     Math       Science      English
#   Tue     History    Math         Art
#   ...
#
# You would call the function like this:
# timetable_to_ics(df, timezone='America/New_York', output_file='my_schedule.ics')


# If you're reading the DataFrame from a CSV file:
# df = pd.read_csv('timetable.csv')
# timetable_to_ics(df)