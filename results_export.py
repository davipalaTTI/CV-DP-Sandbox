"""
Results Export Module - Simplified Event Log Export

Handles hourly segment export of event logs only:
- Event log data per hourly segment
- Multiple output formats (JSON, CSV, Excel)
- No summaries or aggregations
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import datetime


@dataclass
class ExportConfig:
    export_formats: List[str] = None  # ['json', 'csv', 'excel']

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json', 'csv', 'excel']


class ResultsExporter:
    """Handles export of event logs in multiple formats"""

    def __init__(self, output_folder: str, export_config: Optional[ExportConfig] = None):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.config = export_config or ExportConfig()
        self.logger = logging.getLogger(__name__)

        # Create segments folder for hourly exports
        self.segments_folder = self.output_folder / "segments"
        self.segments_folder.mkdir(exist_ok=True)

        self.logger.info(f"Results exporter initialized: {self.output_folder}")

    def export_segment_results(self, segment_id: int, counts: Dict, events: Dict, stats: Any) -> Dict[str, str]:
        """
        Export event log for a single hourly segment

        Args:
            segment_id: Segment identifier
            counts: Current counting data (unused in simplified version)
            events: Events data containing the event list
            stats: Processing statistics (unused in simplified version)

        Returns:
            Dictionary of exported file paths
        """
        try:
            # Extract the event list - it's under 'events' key, not 'event_log'
            event_list = events.get('events', [])

            # Get window times for filename if available
            window_start = events.get('_window_start', datetime.datetime.now().isoformat())
            window_end = events.get('_window_end', datetime.datetime.now().isoformat())

            # Create time string for filename
            try:
                start_dt = datetime.datetime.fromisoformat(window_start.replace('Z', ''))
                end_dt = datetime.datetime.fromisoformat(window_end.replace('Z', ''))
                time_str = f"{start_dt.strftime('%H%M')}-{end_dt.strftime('%H%M')}_{start_dt.strftime('%Y%m%d')}"
            except:
                # Fallback to current time if parsing fails
                timestamp = datetime.datetime.now()
                time_str = timestamp.strftime("%H%M_%Y%m%d")

            # Export in requested formats
            exported_files = {}

            if 'json' in self.config.export_formats:
                json_path = self._export_segment_json(time_str, event_list, window_start, window_end)
                exported_files['json'] = str(json_path)

            if 'csv' in self.config.export_formats:
                csv_path = self._export_segment_csv(time_str, event_list, window_start, window_end)
                exported_files['csv'] = str(csv_path)

            if 'excel' in self.config.export_formats:
                excel_path = self._export_segment_excel(time_str, event_list, window_start, window_end)
                exported_files['excel'] = str(excel_path)

            self.logger.info(f"Segment {segment_id} event log exported to {len(exported_files)} format(s)")
            return exported_files

        except Exception as e:
            self.logger.error(f"Failed to export segment {segment_id} results: {e}")
            return {}

    def _export_segment_json(self, time_str: str, event_list: List[Dict], window_start: str, window_end: str) -> Path:
        """Export event log as JSON"""
        filename = f"events_{time_str}.json"
        filepath = self.segments_folder / filename

        # Filter events to only include desired fields
        filtered_events = []
        for event in event_list:
            # Determine event type
            event_type = 'line_crossing' if 'line_name' in event else 'zone_entry' if 'zone_name' in event else 'unknown'

            filtered_event = {
                'actual_datetime': event.get('actual_datetime', ''),
                'event_type': event_type,
                'track_id': event.get('track_id', ''),
                'class_id': event.get('class_id', ''),
                'class_name': event.get('class_name', ''),
                'line_name': event.get('line_name', ''),
                'zone_name': event.get('zone_name', ''),
                'direction': event.get('direction', ''),
                'confidence': event.get('confidence', ''),
                'speed': event.get('speed', 0.0),
                'speed_units': event.get('speed_units', ''),
                'dwell_seconds': event.get('dwell_seconds', 0.0),
            }
            filtered_events.append(filtered_event)

        # Structure the JSON with metadata and filtered events
        data = {
            "event_count": len(filtered_events),
            "events": filtered_events
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def _export_segment_csv(self, time_str: str, event_list: List[Dict], window_start: str, window_end: str) -> Path:
        """Export event log as CSV"""
        filename = f"events_{time_str}.csv"
        filepath = self.segments_folder / filename

        if not event_list:
            # Write empty CSV with headers
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['actual_datetime', 'event_type', 'track_id',
                               'class_id', 'class_name', 'line_name', 'zone_name',
                               'direction', 'confidence',
                               'speed', 'speed_units', 'dwell_seconds'])
            return filepath

        # Flatten event data for CSV
        rows = []
        for event in event_list:
            # Determine event type
            event_type = 'line_crossing' if 'line_name' in event else 'zone_entry' if 'zone_name' in event else 'unknown'

            # Extract position coordinates if available
            position = event.get('position', None)
            pos_x = position[0] if position else ''
            pos_y = position[1] if position else ''

            row = {
                'actual_datetime': event.get('actual_datetime', ''),
                'event_type': event_type,
                'track_id': event.get('track_id', ''),
                'class_id': event.get('class_id', ''),
                'class_name': event.get('class_name', ''),
                'line_name': event.get('line_name', ''),
                'zone_name': event.get('zone_name', ''),
                'direction': event.get('direction', ''),
                'confidence': event.get('confidence', ''),
                'speed': event.get('speed', 0.0),
                'speed_units': event.get('speed_units', ''),
                'dwell_seconds': event.get('dwell_seconds', 0.0),
            }
            rows.append(row)

        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        else:
            # Write empty CSV with headers
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(rows[0].keys()) if rows else [])

        return filepath

    def _export_segment_excel(self, time_str: str, event_list: List[Dict], window_start: str, window_end: str) -> Path:
        """Export event log as Excel file"""
        filename = f"events_{time_str}.xlsx"
        filepath = self.segments_folder / filename

        if not event_list:
            # Create empty Excel with headers
            df = pd.DataFrame(columns=['actual_datetime', 'event_type', 'track_id',
                                      'class_id', 'class_name', 'line_name', 'zone_name',
                                      'direction', 'confidence',
                                      'speed', 'speed_units', 'dwell_seconds'])
            df.to_excel(filepath, index=False, engine='openpyxl')
            return filepath

        # Flatten event data for Excel
        rows = []
        for event in event_list:
            # Determine event type
            event_type = 'line_crossing' if 'line_name' in event else 'zone_entry' if 'zone_name' in event else 'unknown'

            # Extract position coordinates if available
            position = event.get('position', None)
            pos_x = position[0] if position else ''
            pos_y = position[1] if position else ''

            row = {
                'actual_datetime': event.get('actual_datetime', ''),
                'event_type': event_type,
                'track_id': event.get('track_id', ''),
                'class_id': event.get('class_id', ''),
                'class_name': event.get('class_name', ''),
                'line_name': event.get('line_name', ''),
                'zone_name': event.get('zone_name', ''),
                'direction': event.get('direction', ''),
                'confidence': event.get('confidence', ''),
                'speed': event.get('speed', 0.0),
                'speed_units': event.get('speed_units', ''),
                'dwell_seconds': event.get('dwell_seconds', 0.0),
            }
            rows.append(row)

        # Create DataFrame and export to Excel
        df = pd.DataFrame(rows)

        # Write to Excel with formatting
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Events', index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets['Events']
            for idx, col in enumerate(df.columns):
                if idx < 26:  # Excel column limit for single letters
                    max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)

        return filepath

    def export_final_summary(self, results: Dict) -> Dict[str, str]:
        """
        Export final results - simplified to just combine all event logs

        Args:
            results: Final processing results

        Returns:
            Dictionary of exported file paths
        """
        try:
            exported_files = {}

            # Create a simple final export with all events combined
            all_events = []

            # Extract events from results if available
            if 'events' in results and 'events' in results['events']:
                all_events = results['events']['events']
            elif 'events' in results:
                all_events = results.get('events', [])

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if 'json' in self.config.export_formats:
                filename = f"all_events_{timestamp}.json"
                filepath = self.output_folder / filename

                # Filter events to only include desired fields
                filtered_events = []
                for event in all_events:
                    # Determine event type
                    event_type = 'line_crossing' if 'line_name' in event else 'zone_entry' if 'zone_name' in event else 'unknown'

                    filtered_event = {
                        'actual_datetime': event.get('actual_datetime', ''),
                        'event_type': event_type,
                        'track_id': event.get('track_id', ''),
                        'class_id': event.get('class_id', ''),
                        'class_name': event.get('class_name', ''),
                        'line_name': event.get('line_name', ''),
                        'zone_name': event.get('zone_name', ''),
                        'direction': event.get('direction', ''),
                        'confidence': event.get('confidence', ''),
                        'speed': event.get('speed', 0.0),
                        'speed_units': event.get('speed_units', ''),
                        'dwell_seconds': event.get('dwell_seconds', 0.0),
                    }
                    filtered_events.append(filtered_event)

                data = {
                    "export_timestamp": datetime.datetime.now().isoformat(),
                    "total_events": len(filtered_events),
                    "events": filtered_events
                }

            if 'csv' in self.config.export_formats and all_events:
                filename = f"all_events_{timestamp}.csv"
                filepath = self.output_folder / filename

                # Convert events to DataFrame and save
                df = pd.DataFrame(all_events)
                df.to_csv(filepath, index=False)

                exported_files['final_csv'] = str(filepath)
                self.logger.info(f"Final CSV exported: {filepath}")

            if 'excel' in self.config.export_formats and all_events:
                filename = f"all_events_{timestamp}.xlsx"
                filepath = self.output_folder / filename

                # Convert events to DataFrame and save
                df = pd.DataFrame(all_events)
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='All_Events', index=False)

                    # Auto-adjust column widths
                    worksheet = writer.sheets['All_Events']
                    for idx, col in enumerate(df.columns):
                        if idx < 26:  # Excel column limit
                            max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                            worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)

                exported_files['final_excel'] = str(filepath)
                self.logger.info(f"Final Excel exported: {filepath}")

            return exported_files

        except Exception as e:
            self.logger.error(f"Failed to export final summary: {e}")
            return {}

    def export_live_stats(self, stats_data: Dict) -> str:
        """
        Export current live statistics (kept for compatibility but simplified)
        Just exports the raw stats data without processing
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_stats_{timestamp}.json"
        filepath = self.output_folder / filename

        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2, default=str)

        self.logger.info(f"Live stats exported: {filepath}")
        return str(filepath)

    def get_export_summary(self) -> Dict:
        """Get summary of exported files location"""
        return {
            "output_folder": str(self.output_folder),
            "segments_folder": str(self.segments_folder),
            "export_formats": self.config.export_formats
        }