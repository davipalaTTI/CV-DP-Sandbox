"""
Results Export Module - Simplified Event Log Export

Handles hourly segment export of event logs only:
- Event log data per hourly segment
- Multiple output formats (JSON, CSV, Excel)
- Master event log that appends all events to single Excel file
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
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows


@dataclass
class ExportConfig:
    export_formats: List[str] = None  # ['json', 'csv', 'excel']
    enable_master_log: bool = True  # Enable master event log

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

        # Master event log file
        self.master_log_path = self.output_folder / "master_event_log.xlsx"

        # Initialize master log if it doesn't exist
        if self.config.enable_master_log:
            self._initialize_master_log()

        self.logger.info(f"Results exporter initialized: {self.output_folder}")
        if self.config.enable_master_log:
            self.logger.info(f"Master event log: {self.master_log_path}")

    def _initialize_master_log(self):
        """Initialize the master event log Excel file if it doesn't exist"""
        if not self.master_log_path.exists():
            # Create new file with headers
            df = pd.DataFrame(columns=[
                'actual_datetime', 'event_type', 'track_id',
                'class_id', 'class_name', 'line_name', 'zone_name',
                'direction', 'confidence', 'speed', 'speed_units', 'dwell_seconds',
                'video_source', 'segment_id'
            ])
            df.to_excel(self.master_log_path, sheet_name='Events', index=False, engine='openpyxl')
            self.logger.info(f"Created master event log: {self.master_log_path}")

    def _append_to_master_log(self, event_list: List[Dict], video_source: str = None, segment_id: int = None):
        """
        Append events to the master event log Excel file

        Args:
            event_list: List of event dictionaries to append
            video_source: Optional source video filename
            segment_id: Optional segment identifier
        """
        if not self.config.enable_master_log or not event_list:
            return

        try:
            # Prepare rows to append
            rows = []
            for event in event_list:
                # Determine event type
                event_type = 'line_crossing' if 'line_name' in event else 'zone_entry' if 'zone_name' in event else 'unknown'

                # Sanitize dwell_seconds - set to None/empty for invalid values
                dwell_val = self._sanitize_dwell_time(event.get('dwell_seconds', 0.0))

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
                    'dwell_seconds': dwell_val,  # Will be None for invalid values
                    'video_source': video_source or '',
                    'segment_id': segment_id if segment_id is not None else ''
                }
                rows.append(row)

            # Convert to DataFrame
            new_df = pd.DataFrame(rows)

            # Read existing data
            try:
                existing_df = pd.read_excel(self.master_log_path, sheet_name='Events', engine='openpyxl')
            except:
                # If file doesn't exist or is corrupted, create new
                existing_df = pd.DataFrame(columns=new_df.columns)

            # Append new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Write back to Excel
            with pd.ExcelWriter(self.master_log_path, engine='openpyxl', mode='w') as writer:
                combined_df.to_excel(writer, sheet_name='Events', index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets['Events']
                for idx, col in enumerate(combined_df.columns):
                    if idx < 26:  # Excel column limit for single letters
                        max_len = max(combined_df[col].astype(str).apply(len).max(), len(col)) + 2
                        worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)

            self.logger.info(f"Appended {len(rows)} events to master log (total: {len(combined_df)})")

        except Exception as e:
            self.logger.error(f"Failed to append to master log: {e}")

    def _sanitize_dwell_time(self, dwell_value) -> Optional[float]:
        """
        Validate and sanitize dwell time values.
        Returns None for invalid values (negative or unreasonably large).
        """
        try:
            dwell = float(dwell_value) if dwell_value is not None else 0.0
            # Invalid if negative or greater than 24 hours (86400 seconds)
            if dwell < 0 or dwell > 86400:
                return None
            return round(dwell, 2)
        except (ValueError, TypeError):
            return None

    def export_segment(self, events_list: List, segment_id: int, 
                       start_dt: datetime.datetime, end_dt: datetime.datetime,
                       source_name: str = None) -> Dict[str, str]:
        """
        Export a segment of events - convenience wrapper for VideoWorker
        
        Args:
            events_list: List of CountingEvent objects or event dictionaries
            segment_id: Segment identifier
            start_dt: Segment start datetime
            end_dt: Segment end datetime
            source_name: Optional source video filename
            
        Returns:
            Dictionary of exported file paths
        """
        # Convert CountingEvent objects to dictionaries if needed
        event_dicts = []
        for evt in events_list:
            if hasattr(evt, 'to_dict'):
                d = evt.to_dict()
                # Sanitize dwell time
                d['dwell_seconds'] = self._sanitize_dwell_time(d.get('dwell_seconds', 0.0))
                event_dicts.append(d)
            elif hasattr(evt, '__dataclass_fields__'):
                # Dataclass - convert to dict
                from dataclasses import asdict
                d = asdict(evt)
                # Convert datetime to string if present
                if 'actual_datetime' in d and d['actual_datetime']:
                    if hasattr(d['actual_datetime'], 'isoformat'):
                        d['actual_datetime'] = d['actual_datetime'].isoformat()
                # Sanitize dwell time
                d['dwell_seconds'] = self._sanitize_dwell_time(d.get('dwell_seconds', 0.0))
                event_dicts.append(d)
            elif isinstance(evt, dict):
                evt['dwell_seconds'] = self._sanitize_dwell_time(evt.get('dwell_seconds', 0.0))
                event_dicts.append(evt)
            else:
                # Try to extract attributes
                dwell = self._sanitize_dwell_time(getattr(evt, 'dwell_seconds', 0.0))
                event_dicts.append({
                    'track_id': getattr(evt, 'track_id', ''),
                    'class_id': getattr(evt, 'class_id', ''),
                    'class_name': getattr(evt, 'class_name', ''),
                    'line_name': getattr(evt, 'line_name', ''),
                    'zone_name': getattr(evt, 'zone_name', ''),
                    'direction': getattr(evt, 'direction', ''),
                    'confidence': getattr(evt, 'confidence', ''),
                    'speed': getattr(evt, 'avg_speed', 0.0),
                    'speed_units': getattr(evt, 'speed_units', ''),
                    'dwell_seconds': dwell,
                    'actual_datetime': getattr(evt, 'actual_datetime', '').isoformat() 
                        if hasattr(getattr(evt, 'actual_datetime', ''), 'isoformat') 
                        else str(getattr(evt, 'actual_datetime', ''))
                })
        
        # Build the events dictionary in the format expected by export_segment_results
        events_dict = {
            'events': event_dicts,
            '_window_start': start_dt.isoformat() if start_dt else datetime.datetime.now().isoformat(),
            '_window_end': end_dt.isoformat() if end_dt else datetime.datetime.now().isoformat()
        }
        
        return self.export_segment_results(
            segment_id=segment_id,
            counts={},  # Not used in simplified version
            events=events_dict,
            stats=None,  # Not used in simplified version
            video_source=source_name
        )

    def export_video_summary(self, results: Dict, video_name: str = None) -> Dict[str, str]:
        """
        Export summary for a single video - convenience wrapper for VideoWorker
        
        Args:
            results: Dictionary with 'final_counts', 'events_summary', 'stats' keys
            video_name: Name of the video file
            
        Returns:
            Dictionary of exported file paths
        """
        try:
            exported_files = {}
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"{video_name}_" if video_name else ""
            
            # Create summaries folder
            summaries_folder = self.output_folder / "summaries"
            summaries_folder.mkdir(parents=True, exist_ok=True)
            
            # Export as JSON
            if 'json' in self.config.export_formats:
                filename = f"{prefix}summary_{timestamp}.json"
                filepath = summaries_folder / filename
                
                # Build summary data
                summary_data = {
                    'video_name': video_name,
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'final_counts': results.get('final_counts', {}),
                    'events_summary': results.get('events_summary', {}),
                    'stats': {}
                }
                
                # Handle ProcessingStats dataclass
                stats = results.get('stats')
                if stats:
                    if hasattr(stats, '__dataclass_fields__'):
                        from dataclasses import asdict
                        summary_data['stats'] = asdict(stats)
                    elif hasattr(stats, '__dict__'):
                        summary_data['stats'] = vars(stats)
                    elif isinstance(stats, dict):
                        summary_data['stats'] = stats
                
                with open(filepath, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)
                
                exported_files['summary_json'] = str(filepath)
                self.logger.info(f"Video summary exported: {filepath}")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export video summary: {e}")
            return {}

    def export_segment_results(self, segment_id: int, counts: Dict, events: Dict, stats: Any,
                              video_source: str = None) -> Dict[str, str]:
        """
        Export event log for a single hourly segment

        Args:
            segment_id: Segment identifier
            counts: Current counting data (unused in simplified version)
            events: Events data containing the event list
            stats: Processing statistics (unused in simplified version)
            video_source: Optional source video filename

        Returns:
            Dictionary of exported file paths
        """
        try:
            # Extract the event list - it's under 'events' key, not 'event_log'
            event_list = events.get('events', [])

            # Append to master log FIRST
            if self.config.enable_master_log:
                self._append_to_master_log(event_list, video_source=video_source, segment_id=segment_id)

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
                'dwell_seconds': self._sanitize_dwell_time(event.get('dwell_seconds', 0.0)),
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
        summary = {
            "output_folder": str(self.output_folder),
            "segments_folder": str(self.segments_folder),
            "export_formats": self.config.export_formats
        }

        if self.config.enable_master_log:
            summary["master_log"] = str(self.master_log_path)

            # Get event count from master log
            try:
                df = pd.read_excel(self.master_log_path, sheet_name='Events', engine='openpyxl')
                summary["master_log_event_count"] = len(df)
            except:
                summary["master_log_event_count"] = 0

        return summary