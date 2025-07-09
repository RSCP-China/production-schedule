import streamlit as st

# Translation dictionary update - add new translations
TRANSLATIONS = {
    'en': {
        'time': 'Time',
        'date': 'Date',
        'utilization': 'Utilization',
        'work_center': 'Work Center',
        'schedule_results': 'Schedule Results',
        'max_batch_hours': 'Maximum Batch Hours',
        'max_batch_help': 'Maximum allowed hours per batch to prevent excessive batch sizes.',
        'time_window': 'Time Window (Days)',
        'time_window_help': 'Orders with same part number within this time window will be batched',
        'generate_schedule': 'Generate Schedule',
        'generating_schedule': 'Generating schedule...',
        'no_operations': 'No operations could be scheduled. Please check resource availability.',
        'schedule_warnings': 'Show Scheduling Warnings',
        'schedule_generated': 'Schedule generated!',
        'download_schedule': 'Download Schedule',
        'error_scheduling': 'Error during scheduling: ',
        'schedule_statistics': 'Schedule Statistics',
        'original_orders': 'Original orders',
        'scheduled_orders': 'Scheduled orders',
        'avg_run_time': 'Average run time',
        'schedule_performance': 'Schedule Performance',
        'total_makespan': 'Total makespan',
        'delayed_jobs': 'Delayed jobs',
        'avg_delay': 'Average delay',
        'hours': 'hours',
        'visualizations': 'Visualizations',
        'overview': 'Overview',
        'gantt_chart': 'Gantt Chart',
        'work_center_utilization': 'Work Center Utilization',
        'select_job': 'Select a job to highlight its operations',
        'all_jobs': 'All Jobs',
        'job_details': 'Job Details',
        'part_number': 'Part Number',
        'num_operations': 'Number of Operations',
        'total_processing': 'Total Processing Time',
        'total_setup': 'Total Setup Time',
        'total_duration': 'Total Duration',
        'updating_gantt': 'Updating Gantt chart...',
        'production_schedule': 'Production Schedule by Work Center',
        'job_prefix': 'Job',
        'orders_total': 'Total orders',
        'resources_total': 'Total resources'
    },
    'zh': {
        'time': '时间',
        'date': '日期',
        'utilization': '利用率',
        'work_center': '工作中心',
        'schedule_results': '调度结果',
        'max_batch_hours': '最大批次小时数',
        'max_batch_help': '防止批次规模过大的每批次最大允许小时数。',
        'time_window': '时间窗口（天）',
        'time_window_help': '在此时间窗口内具有相同零件编号的订单将被批处理',
        'generate_schedule': '生成调度',
        'generating_schedule': '正在生成调度...',
        'no_operations': '无法安排任何操作。请检查资源可用性。',
        'schedule_warnings': '显示调度警告',
        'schedule_generated': '调度已生成！',
        'download_schedule': '下载调度',
        'error_scheduling': '调度错误: ',
        'schedule_statistics': '调度统计',
        'original_orders': '原始订单',
        'scheduled_orders': '已调度订单',
        'avg_run_time': '平均运行时间',
        'schedule_performance': '调度性能',
        'total_makespan': '总生产时间',
        'delayed_jobs': '延迟工作',
        'avg_delay': '平均延迟',
        'hours': '小时',
        'visualizations': '可视化',
        'overview': '概览',
        'gantt_chart': '甘特图',
        'work_center_utilization': '工作中心利用率',
        'select_job': '选择要突出显示的作业',
        'all_jobs': '所有作业',
        'job_details': '作业详情',
        'part_number': '零件编号',
        'num_operations': '操作数量',
        'total_processing': '总加工时间',
        'total_setup': '总设置时间',
        'total_duration': '总持续时间',
        'updating_gantt': '更新甘特图...',
        'production_schedule': '按工作中心的生产调度',
        'job_prefix': '作业',
        'orders_total': '订单总数',
        'resources_total': '资源总数'
    }
}
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta, time
from pathlib import Path

# Translation dictionary
TRANSLATIONS = {
    'en': {
        'app_title': 'Production Scheduler',
        'optimization_weights': 'Optimization Weights',
        'weights_info': 'Allocate weights to different optimization strategies. The sum must equal 100%',
        'minimize_makespan': 'Minimize Total Makespan (%)',
        'prioritize_due_dates': 'Prioritize Due Dates (%)',
        'maximize_utilization': 'Maximize Resource Utilization (%)',
        'minimize_setup': 'Minimize Setup Times (%)',
        'total': 'Total',
        'weights_error': 'Weights must sum to 100%',
        'batching_config': 'Batching Configuration',
        'max_batch_hours': 'Maximum Batch Hours',
        'upload_orders': 'Upload Production Orders (CSV)',
        'upload_resources': 'Upload Resources Data (CSV)',
        'orders_preview': 'Production Orders Preview:',
        'total_orders': 'Total orders',
        'resources_preview': 'Resources Preview:',
        'total_resources': 'Total resources',
    },
    'zh': {
        'app_title': '生产调度系统',
        'optimization_weights': '优化权重',
        'weights_info': '分配不同优化策略的权重。总和必须等于100%',
        'minimize_makespan': '最小化总生产时间 (%)',
        'prioritize_due_dates': '优先考虑交付日期 (%)',
        'maximize_utilization': '最大化资源利用率 (%)',
        'minimize_setup': '最小化设置时间 (%)',
        'total': '总计',
        'weights_error': '权重总和必须为100%',
        'batching_config': '批次配置',
        'max_batch_hours': '最大批次小时数',
        'upload_orders': '上传生产订单 (CSV)',
        'upload_resources': '上传资源数据 (CSV)',
        'orders_preview': '生产订单预览:',
        'total_orders': '订单总数',
        'resources_preview': '资源预览:',
        'total_resources': '资源总数',
    }
}

def get_text(key):
    """Get text in the current language"""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS[lang].get(key, TRANSLATIONS['en'].get(key, key))


def calculate_batch_hours(batch):
    """Calculate total hours for a batch including all operations"""
    total_hours = 0
    for op_data in batch['operations'].values():
        total_hours += op_data['Run Time'] + op_data['Setup Time']
    return total_hours

def validate_weights(weights):
    """Validate that weights sum to 100%"""
    total = sum(weights.values())
    return abs(total - 100) < 0.01

def get_optimization_weights():
    """Get optimization weights and batching configuration"""
    st.sidebar.header(get_text('optimization_weights'))
    st.sidebar.info(get_text('weights_info'))
    
    weights = {
        'makespan': st.sidebar.number_input(get_text('minimize_makespan'),
                                        min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'due_date': st.sidebar.number_input(get_text('prioritize_due_dates'),
                                        min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'utilization': st.sidebar.number_input(get_text('maximize_utilization'),
                                           min_value=0.0, max_value=100.0, value=25.0, step=5.0),
        'setup_time': st.sidebar.number_input(get_text('minimize_setup'),
                                          min_value=0.0, max_value=100.0, value=25.0, step=5.0)
    }
    
    total = sum(weights.values())
    st.sidebar.write(f"{get_text('total')}: {total}%")
    
    if abs(total - 100) > 0.01:
        st.sidebar.error(get_text('weights_error'))
        return None
    
    st.sidebar.markdown("---")
    st.sidebar.header(get_text('batching_config'))
    
    max_batch_hours = st.sidebar.number_input(
        get_text('max_batch_hours'),
        min_value=1,
        max_value=500,
        value=150,
        help=get_text('max_batch_help')
    )
    
    batch_window = st.sidebar.number_input(
        get_text('time_window'),
        min_value=0,
        max_value=30,
        value=5,
        help=get_text('time_window_help')
    )
    
    result = {k: v/100.0 for k, v in weights.items()}
    result['batch_window'] = batch_window
    result['max_batch_hours'] = max_batch_hours
    return result

def load_production_orders(file):
    """Load and validate production orders CSV file"""
    try:
        df = None
        error_messages = []
        
        # Try different encodings
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding, on_bad_lines='skip')
                if not df.empty:
                    break
            except UnicodeDecodeError:
                error_messages.append(f"Failed to decode with {encoding} encoding")
                continue
            except pd.errors.EmptyDataError:
                error_messages.append(f"Empty file or no data found with {encoding} encoding")
                continue
            except Exception as e:
                error_messages.append(f"Error with {encoding} encoding: {str(e)}")
                continue
        
        if df is None or df.empty:
            raise ValueError(f"Could not load file with any encoding. Errors: {'; '.join(error_messages)}")
        
        # Print column names for debugging
        print("Loaded columns:", df.columns.tolist())
        print("Preview of first row:", df.iloc[0].to_dict() if not df.empty else "No data")
        
        # Check required columns
        required_columns = ['Job Number', 'Part Number', 'Due Date', 'operation sequence',
                          'WorkCenter', 'Place', 'Run Time', 'Setup Time', 'JobPriority']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}\n"
                           f"Available columns are: {', '.join(df.columns)}")
        
        # Convert Due Date to datetime
        df['Due Date'] = pd.to_datetime(df['Due Date'].replace('#VALUE!', pd.NaT), format='%d/%m/%Y', errors='coerce')
        
        # Drop rows with invalid dates
        invalid_dates = df[df['Due Date'].isna()]
        if not invalid_dates.empty:
            st.warning(f"Found {len(invalid_dates)} rows with invalid dates. These orders will be skipped.")
            df = df.dropna(subset=['Due Date'])
        
        # Convert numeric columns
        df['Run Time'] = pd.to_numeric(df['Run Time'], errors='coerce')
        df['Setup Time'] = pd.to_numeric(df['Setup Time'], errors='coerce')
        df['JobPriority'] = pd.to_numeric(df['JobPriority'], errors='coerce')
        
        # Drop any rows with invalid numeric values
        df = df.dropna(subset=['Run Time', 'Setup Time', 'JobPriority'])
        
        if df.empty:
            raise ValueError("No valid data remains after processing")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_resources(file):
    """Load and validate resources CSV file"""
    try:
        df = None
        error_messages = []
        
        # Try different encodings
        for encoding in ['gbk', 'gb2312', 'utf-8']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding, on_bad_lines='skip')
                if not df.empty:
                    break
            except UnicodeDecodeError:
                error_messages.append(f"Failed to decode with {encoding} encoding")
                continue
            except pd.errors.EmptyDataError:
                error_messages.append(f"Empty file or no data found with {encoding} encoding")
                continue
            except Exception as e:
                error_messages.append(f"Error with {encoding} encoding: {str(e)}")
                continue
        
        if df is None or df.empty:
            raise ValueError(f"Could not load resources file with any encoding. Errors: {'; '.join(error_messages)}")
            
        # Print column names for debugging
        print("Loaded resource columns:", df.columns.tolist())
        print("Preview of first resource row:", df.iloc[0].to_dict() if not df.empty else "No data")
        
        # Check required columns
        required_columns = ['WorkCenter', 'Place', 'Available Quantity', 'Shift hours']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}\n"
                           f"Available columns are: {', '.join(df.columns)}")
        
        # Convert numeric columns
        df['Available Quantity'] = pd.to_numeric(df['Available Quantity'], errors='coerce')
        df['Shift hours'] = pd.to_numeric(df['Shift hours'], errors='coerce')
        
        # Drop any rows with invalid numeric values
        df = df.dropna(subset=['Available Quantity', 'Shift hours'])
        
        if df.empty:
            raise ValueError("No valid resource data remains after processing")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading resources file: {str(e)}")
        return None

def create_batches(orders_df, max_batch_hours):
    """Create batches of operations optimizing for setup time"""
    batches = []
    
    # Group operations by Part Number and WorkCenter
    grouped_ops = orders_df.groupby(['Part Number', 'WorkCenter'])
    
    for (part_number, work_center), group in grouped_ops:
        # Sort operations by priority and due date
        group = group.sort_values(['JobPriority', 'Due Date'])
        
        current_batch = []
        batch_run_time = 0
        batch_setup_time = 0
        
        for _, operation in group.iterrows():
            # For new batch, consider full setup time
            if not current_batch:
                batch_setup_time = operation['Setup Time']
                batch_run_time = operation['Run Time']
                current_batch.append(operation)
            else:
                # For existing batch, only add run time (setup already counted)
                if batch_run_time + operation['Run Time'] <= max_batch_hours:
                    batch_run_time += operation['Run Time']
                    batch_setup_time = max(batch_setup_time, operation['Setup Time'])
                    current_batch.append(operation)
                else:
                    # Finalize current batch
                    batches.append({
                        'operations': current_batch,
                        'work_center': work_center,
                        'part_number': part_number,
                        'total_hours': batch_run_time + batch_setup_time,
                        'setup_time': batch_setup_time,
                        'priority': min(op['JobPriority'] for op in current_batch)
                    })
                    # Start new batch with current operation
                    current_batch = [operation]
                    batch_run_time = operation['Run Time']
                    batch_setup_time = operation['Setup Time']
        
        # Add final batch
        if current_batch:
            batches.append({
                'operations': current_batch,
                'work_center': work_center,
                'part_number': part_number,
                'total_hours': batch_run_time + batch_setup_time,
                'setup_time': batch_setup_time,
                'priority': min(op['JobPriority'] for op in current_batch)
            })
    
    # Sort batches by priority and due date
    batches.sort(key=lambda x: (x['priority'], min(op['Due Date'] for op in x['operations'])))
    return batches

def schedule_operations(orders_df, resources_df, settings):
    """Schedule operations with optimized batching and priority-based forwarding"""
    today = pd.Timestamp.now().normalize()
    
    # Initialize machine availability tracking
    machine_schedules = {}
    for _, resource in resources_df.iterrows():
        work_center = resource['WorkCenter']
        num_machines = int(resource['Available Quantity'])
        if num_machines > 0:
            machine_schedules[work_center] = [{
                'available_from': today,
                'machine_id': i + 1,
                'total_hours': 0
            } for i in range(num_machines)]

    # Create optimized batches
    batches = create_batches(orders_df, settings['max_batch_hours'])
    
    # Track completion times for each job's operations
    job_completion_times = {}
    scheduled_orders = []

    # Process each batch while respecting job dependencies
    for batch in batches:
        work_center = batch['work_center']
        machines = machine_schedules[work_center]
        
        # Find earliest available machine considering job dependencies
        earliest_start = None
        selected_machine = None
        
        for machine in machines:
            possible_start = machine['available_from']
            
            # Check dependencies for all operations in batch
            for operation in batch['operations']:
                job_number = operation['Job Number']
                op_seq = operation['operation sequence']
                
                # If job has previous operations, consider their completion times
                if job_number in job_completion_times:
                    prev_op_time = job_completion_times[job_number].get(op_seq - 1)
                    if prev_op_time:
                        possible_start = max(possible_start, prev_op_time)
            
            if earliest_start is None or possible_start < earliest_start:
                earliest_start = possible_start
                selected_machine = machine
        
        if selected_machine is None:
            print(f"Warning: No machine available for {work_center}")
            continue
        
        # Schedule all operations in the batch
        current_time = earliest_start
        
        # Apply setup time once for the batch
        setup_end = current_time + pd.Timedelta(hours=batch['setup_time'])
        current_time = setup_end
        
        # Schedule each operation in the batch
        for operation in batch['operations']:
            # Only use run time since setup is already accounted for
            run_time = operation['Run Time']
            end_time = current_time + pd.Timedelta(hours=run_time)
            
            # Record scheduled operation
            scheduled_op = operation.copy()
            scheduled_op['Start Time'] = current_time.strftime('%Y-%m-%d %H:%M')
            scheduled_op['Finish Time'] = end_time.strftime('%Y-%m-%d %H:%M')
            scheduled_op['Machine'] = f"{work_center}_M{selected_machine['machine_id']}"
            scheduled_orders.append(scheduled_op)
            
            # Update job completion tracking
            job_number = operation['Job Number']
            op_seq = operation['operation sequence']
            if job_number not in job_completion_times:
                job_completion_times[job_number] = {}
            job_completion_times[job_number][op_seq] = end_time
            
            # Move to next operation time
            current_time = end_time
        
        # Update machine availability
        selected_machine['available_from'] = current_time
        selected_machine['total_hours'] += batch['total_hours']
    
    result_df = pd.DataFrame(scheduled_orders)
    
    if len(result_df) > 0:
        # Convert datetime strings to datetime objects for calculations
        result_df['Start Time'] = pd.to_datetime(result_df['Start Time'])
        result_df['Finish Time'] = pd.to_datetime(result_df['Finish Time'])
        
        # Calculate schedule metrics
        latest_end = result_df['Finish Time'].max()
        earliest_start = result_df['Start Time'].min()
        makespan = (latest_end - earliest_start).total_seconds() / 3600
        
        print(f"\nSchedule Metrics:")
        print(f"Makespan: {makespan:.1f} hours")
        print(f"Schedule spans from {earliest_start.strftime('%Y-%m-%d')} to {latest_end.strftime('%Y-%m-%d')}")
        
        # Calculate and print batching metrics
        total_batches = len(batches)
        avg_batch_size = sum(len(batch['operations']) for batch in batches) / total_batches
        total_setup_time = sum(batch['setup_time'] for batch in batches)
        setup_time_saved = sum(
            sum(op['Setup Time'] for op in batch['operations']) - batch['setup_time']
            for batch in batches
        )
        
        print(f"\nBatching Metrics:")
        print(f"Total batches: {total_batches}")
        print(f"Average batch size: {avg_batch_size:.1f} operations")
        print(f"Total setup time: {total_setup_time:.1f} hours")
        print(f"Setup time saved: {setup_time_saved:.1f} hours")
        
        # Convert back to string format for display
        result_df['Start Time'] = result_df['Start Time'].dt.strftime('%Y-%m-%d %H:%M')
        result_df['Finish Time'] = result_df['Finish Time'].dt.strftime('%Y-%m-%d %H:%M')
    
    return result_df

def create_gantt_chart(df, selected_job=None):
    """Create a Gantt chart for the production schedule with optional job highlighting"""
    df = df.copy()
    
    # Convert times once
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['Finish Time'] = pd.to_datetime(df['Finish Time'])
    
    # Sort by WorkCenter and Start Time
    df_sorted = df.sort_values(['WorkCenter', 'Start Time'])
    
    # Create color map once
    colors = px.colors.qualitative.Set3
    work_centers = df_sorted['WorkCenter'].unique()
    color_map = {wc: colors[i % len(colors)] for i, wc in enumerate(work_centers)}
    
    # Group by WorkCenter to reduce number of traces
    traces = []
    for work_center in work_centers:
        wc_data = df_sorted[df_sorted['WorkCenter'] == work_center]
        
        # Split data based on selection
        if selected_job:
            # Selected jobs
            selected_data = wc_data[wc_data['Job Number'] == selected_job]
            if not selected_data.empty:
                # Create individual bars for each selected operation
                for _, row in selected_data.iterrows():
                    traces.append(go.Bar(
                        base=row['Start Time'],
                        x=[(row['Finish Time'] - row['Start Time']).total_seconds() / 3600],
                        y=[work_center],
                        orientation='h',
                        marker=dict(
                            color='rgb(255, 0, 0)',  # Pure bright red for selected job
                            opacity=1.0,
                            line=dict(width=4, color='black')
                        ),
                        hovertext=f"{get_text('job_prefix')}: {row['Job Number']}<br>"
                                 f"{get_text('part_number')}: {row['Part Number']}<br>"
                                 f"Operation: {row['operation sequence']}<br>"
                                 f"Setup: {row['Setup Time']}h<br>"
                                 f"Run: {row['Run Time']}h<br>"
                                 f"Start: {row['Start Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                                 f"End: {row['Finish Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                                 f"Machine: {row['Machine']}",
                        hoverinfo='text',
                    showlegend=False
                ))
            
            # Non-selected jobs
            other_data = wc_data[wc_data['Job Number'] != selected_job]
            if not other_data.empty:
                # Create individual bars for each non-selected operation
                for _, row in other_data.iterrows():
                    traces.append(go.Bar(
                        base=row['Start Time'],
                        x=[(row['Finish Time'] - row['Start Time']).total_seconds() / 3600],
                        y=[work_center],
                        orientation='h',
                        marker=dict(
                            color='rgb(240, 240, 240)',  # Extremely light grey for non-selected jobs
                            opacity=0.3,
                            line=dict(width=1, color='rgb(200, 200, 200)')  # Light border
                        ),
                        hovertext=f"{get_text('job_prefix')}: {row['Job Number']}<br>"
                                 f"{get_text('part_number')}: {row['Part Number']}<br>"
                                 f"Operation: {row['operation sequence']}<br>"
                                 f"Setup: {row['Setup Time']}h<br>"
                                 f"Run: {row['Run Time']}h<br>"
                                 f"Start: {row['Start Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                                 f"End: {row['Finish Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                                 f"Machine: {row['Machine']}",
                        hoverinfo='text',
                    showlegend=False
                ))
        else:
            # No selection - show all jobs normally
            # No selection - show all jobs normally with individual bars
            for _, row in wc_data.iterrows():
                traces.append(go.Bar(
                    base=row['Start Time'],
                    x=[(row['Finish Time'] - row['Start Time']).total_seconds() / 3600],
                    y=[work_center],
                    orientation='h',
                    marker=dict(
                        color=color_map[work_center],
                        opacity=0.9,
                        line=dict(width=1, color='rgb(96, 96, 96)')  # Subtle border
                    ),
                    hovertext=f"{get_text('job_prefix')}: {row['Job Number']}<br>"
                             f"{get_text('part_number')}: {row['Part Number']}<br>"
                             f"Operation: {row['operation sequence']}<br>"
                             f"Setup: {row['Setup Time']}h<br>"
                             f"Run: {row['Run Time']}h<br>"
                             f"Start: {row['Start Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                             f"End: {row['Finish Time'].strftime('%Y-%m-%d %H:%M')}<br>"
                             f"Machine: {row['Machine']}",
                    hoverinfo='text',
                showlegend=False
            ))
    
    # Create figure with all traces
    fig = go.Figure(data=traces)
    
    # Optimize layout
    layout = {
        'title': get_text('production_schedule'),
        'xaxis_title': get_text('time'),
        'yaxis_title': "Work Center",
        'height': 400 + len(work_centers) * 40,
        'barmode': 'overlay',
        'bargap': 0.2,
        'xaxis': {
            'rangeslider': {'visible': True},
            'type': 'date',
            'automargin': True
        },
        'yaxis': {
            'tickmode': 'linear',
            'type': 'category',
            'automargin': True
        },
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
        'showlegend': False,
        'hovermode': 'closest'
    }
    fig.update_layout(layout)
    
    return fig

def create_workload_heatmap(orders_df, resources_df):
    """Create a heatmap showing work center utilization across dates"""
    orders_df = orders_df.copy()
    orders_df['Start Time'] = pd.to_datetime(orders_df['Start Time'])
    orders_df['Finish Time'] = pd.to_datetime(orders_df['Finish Time'])
    
    # Create a date range covering the entire schedule
    date_range = pd.date_range(
        start=orders_df['Start Time'].min().normalize(),
        end=orders_df['Finish Time'].max().normalize(),
        freq='D'
    )
    
    # Initialize workload calculation
    workload_data = []
    
    # Process each operation
    for _, operation in orders_df.iterrows():
        # Get the operation dates
        op_dates = pd.date_range(
            start=operation['Start Time'].normalize(),
            end=operation['Finish Time'].normalize(),
            freq='D'
        )
        
        # Calculate hours per day for this operation
        for op_date in op_dates:
            start = max(operation['Start Time'], pd.Timestamp(op_date))
            end = min(operation['Finish Time'], pd.Timestamp(op_date) + pd.Timedelta(days=1))
            hours = (end - start).total_seconds() / 3600
            
            workload_data.append({
                'WorkCenter': operation['WorkCenter'],
                'Date': op_date,
                'Hours': hours
            })
    
    # Convert to DataFrame
    workload = pd.DataFrame(workload_data)
    
    # Sum hours by work center and date
    workload = workload.groupby(['WorkCenter', 'Date'])['Hours'].sum().reset_index()
    
    # Create full date range for all work centers
    all_workcenters = orders_df['WorkCenter'].unique()
    full_index = pd.MultiIndex.from_product(
        [all_workcenters, date_range],
        names=['WorkCenter', 'Date']
    )
    
    # Reindex to include all dates, fill missing values with 0
    workload = workload.set_index(['WorkCenter', 'Date']).reindex(full_index, fill_value=0).reset_index()
    
    # Merge with resources to get shift hours and number of machines
    workload = pd.merge(
        workload,
        resources_df[['WorkCenter', 'Shift hours', 'Available Quantity']],
        on='WorkCenter',
        how='left'
    )
    
    # Calculate utilization percentage (hours used / total available hours per day)
    workload['Total Available Hours'] = workload['Shift hours'] * workload['Available Quantity']
    workload['Utilization'] = (workload['Hours'] / workload['Total Available Hours'] * 100).clip(0, 100)
    
    # Pivot data for heatmap
    workload_pivot = workload.pivot_table(
        index='WorkCenter',
        columns='Date',
        values='Utilization',
        fill_value=0
    )
    
    # Sort work centers
    workload_pivot = workload_pivot.reindex(sorted(workload_pivot.index))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=workload_pivot.values.tolist(),
        x=workload_pivot.columns.strftime('%Y-%m-%d'),
        y=workload_pivot.index,
        colorscale=[
            [0, 'lightblue'],     # 0% utilization
            [0.4, 'lightgreen'],  # 40% utilization
            [0.7, 'yellow'],      # 70% utilization
            [0.9, 'orange'],      # 90% utilization
            [1.0, 'red']          # 100% utilization
        ],
        text=[[f"{val:.1f}%" for val in row] for row in workload_pivot.values],
        texttemplate="%{text}",
        hovertemplate=f"{get_text('work_center')}: %{{y}}<br>{get_text('date')}: %{{x}}<br>{get_text('utilization')}: %{{z:.1f}}%<br>",
        colorbar=dict(
            title='Utilization %',
            ticksuffix='%'
        ),
        zmin=0,
        zmax=100
    ))
    
    # Update layout
    fig.update_layout(
        title="Work Center Utilization Heatmap",
        xaxis_title="Date",
        yaxis_title="Work Center",
        height=400 + len(workload_pivot.index) * 40,
        yaxis=dict(
            tickmode='linear',
            type='category'
        ),
        margin=dict(t=50, l=200)
    )
    
    return fig

def main():
    # Set page config at the beginning
    st.set_page_config(page_title="Production Scheduler", layout="wide")
    
    # Initialize all session state variables at start
    if 'complete_scheduled_orders' not in st.session_state:
        st.session_state['complete_scheduled_orders'] = None
    if 'schedule_generated' not in st.session_state:
        st.session_state['schedule_generated'] = False
    if 'resources_df' not in st.session_state:
        st.session_state['resources_df'] = None
    if 'language' not in st.session_state:
        st.session_state['language'] = 'en'
    
    # Language selector
    selected_lang = st.sidebar.selectbox(
        "Language / 语言",
        ["English", "中文"],
        index=0 if st.session_state['language'] == 'en' else 1
    )
    st.session_state['language'] = 'en' if selected_lang == 'English' else 'zh'
    
    st.title(get_text('app_title'))
    
    # Get optimization weights and batch settings
    settings = get_optimization_weights()
    if not settings:
        return
    
    col1, col2 = st.columns(2)
    with col1:
        orders_file = st.file_uploader(get_text('upload_orders'), type=['csv'])
        if orders_file:
            orders_df = load_production_orders(orders_file)
            if orders_df is not None:
                st.write(get_text('orders_preview'))
                st.dataframe(orders_df.head(), height=200)
                st.info(f"{get_text('total_orders')}: {len(orders_df)}")
                
    with col2:
        resources_file = st.file_uploader(get_text('upload_resources'), type=['csv'])
        if resources_file:
            resources_df = load_resources(resources_file)
            if resources_df is not None:
                st.session_state['resources_df'] = resources_df
                st.write(get_text('resources_preview'))
                st.dataframe(resources_df.head(), height=200)
                st.info(f"{get_text('total_resources')}: {len(resources_df)}")

    if orders_file and resources_file:
        orders_df = load_production_orders(orders_file)
        resources_df = st.session_state['resources_df']
        
        if orders_df is not None and resources_df is not None:
            generate_schedule = st.button(get_text('generate_schedule'))
            if generate_schedule:
                try:
                    with st.spinner(get_text('generating_schedule')):
                        # Capture warnings in a StringIO buffer
                        import io
                        import sys
                        warning_output = io.StringIO()
                        sys.stdout = warning_output
                        
                        # Apply scheduling with finite capacity and batching configuration
                        scheduled_orders = schedule_operations(orders_df, resources_df, settings)
                        st.session_state['complete_scheduled_orders'] = scheduled_orders.copy()
                        st.session_state['schedule_generated'] = True
                        
                        # Restore stdout and get warnings
                        sys.stdout = sys.__stdout__
                        warnings = warning_output.getvalue()
                        
                        if scheduled_orders is None or len(scheduled_orders) == 0:
                            st.error(get_text('no_operations'))
                            return
                        
                        # Display any warnings that occurred during scheduling
                        if warnings.strip():
                            with st.expander(get_text('schedule_warnings'), expanded=False):
                                st.warning(warnings)
                                
                        st.success(get_text('schedule_generated'))
                        
                except Exception as e:
                    st.error(f"{get_text('error_scheduling')}{str(e)}")
                    return
                
                st.success(get_text('schedule_generated'))
                
                # Calculate and display metrics
                original_orders = len(orders_df['Job Number'].unique())
                scheduled_orders_count = len(scheduled_orders['Job Number'].unique())
                avg_run_time = scheduled_orders['Run Time'].mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{get_text('schedule_statistics')}:**")
                    st.write(f"{get_text('original_orders')}: {original_orders}")
                    st.write(f"{get_text('scheduled_orders')}: {scheduled_orders_count}")
                    st.write(f"{get_text('avg_run_time')}: {avg_run_time:.1f} {get_text('hours')}")
                
                with col2:
                    # Calculate schedule performance metrics
                    scheduled_orders['Start Time'] = pd.to_datetime(scheduled_orders['Start Time'])
                    scheduled_orders['Finish Time'] = pd.to_datetime(scheduled_orders['Finish Time'])
                    scheduled_orders['Due Date'] = pd.to_datetime(scheduled_orders['Due Date'])
                    
                    makespan = (scheduled_orders['Finish Time'].max() -
                              scheduled_orders['Start Time'].min()).total_seconds() / 3600
                    
                    delayed_jobs = scheduled_orders[scheduled_orders['Finish Time'] > scheduled_orders['Due Date']]
                    if len(delayed_jobs) > 0:
                        avg_delay = (delayed_jobs['Finish Time'] - delayed_jobs['Due Date']).mean().total_seconds() / 3600
                    else:
                        avg_delay = 0
                    
                    st.write(f"**{get_text('schedule_performance')}:**")
                    st.write(f"{get_text('total_makespan')}: {makespan:.1f} {get_text('hours')}")
                    st.write(f"{get_text('delayed_jobs')}: {len(delayed_jobs)}")
                    st.write(f"{get_text('avg_delay')}: {avg_delay:.1f} {get_text('hours')}")
                
                # Display schedule results first
                st.subheader(get_text('schedule_results'))
                display_cols = ['Job Number', 'Part Number', 'Due Date',
                              'JobPriority', 'operation sequence', 'Quantity', 'WorkCenter',
                              'Run Time', 'Setup Time', 'Place', 'Customer', 'Machine',
                              'Start Time', 'Finish Time']
                st.dataframe(scheduled_orders[display_cols].sort_values(['Start Time']), use_container_width=True)

                # Store results in session state
                st.session_state['complete_scheduled_orders'] = scheduled_orders.copy()
                st.session_state['schedule_generated'] = True

    # Display visualizations if schedule has been generated
    if st.session_state.get('schedule_generated', False) and st.session_state.get('complete_scheduled_orders') is not None:
        # Create tabs for visualizations
        st.subheader(get_text('visualizations'))
        tab1, tab2 = st.tabs([get_text('overview'), get_text('gantt_chart')])
        
        with tab1:
            # Create heatmap in the first tab
            st.subheader(get_text('work_center_utilization'))
            if st.session_state['resources_df'] is not None:
                fig_heatmap = create_workload_heatmap(st.session_state['complete_scheduled_orders'], st.session_state['resources_df'])
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            try:
                # Add job selection dropdown
                unique_jobs = sorted(st.session_state['complete_scheduled_orders']['Job Number'].unique())
                job_col1, job_col2 = st.columns([2, 1])
                
                with job_col1:
                    selected_job = st.selectbox(
                        get_text('select_job'),
                        [get_text('all_jobs')] + list(unique_jobs),
                        format_func=lambda x: f"{get_text('job_prefix')} {x}" if x != get_text('all_jobs') else x,
                        key='job_selector'
                    )
                
                # Convert selection to actual job number
                job_to_highlight = None if selected_job == get_text('all_jobs') else selected_job
                
                # Ensure datetime columns are properly formatted
                df_for_gantt = st.session_state['complete_scheduled_orders'].copy()
                df_for_gantt['Start Time'] = pd.to_datetime(df_for_gantt['Start Time'])
                df_for_gantt['Finish Time'] = pd.to_datetime(df_for_gantt['Finish Time'])
                
                # Show job details if a specific job is selected
                if job_to_highlight:
                    job_ops = df_for_gantt[df_for_gantt['Job Number'] == job_to_highlight].copy()
                    total_duration = (job_ops['Finish Time'].max() - job_ops['Start Time'].min()).total_seconds() / 3600
                    
                    with job_col2:
                        if not job_ops.empty:
                            st.write(f"**{get_text('job_details')}:**")
                            st.write(f"{get_text('part_number')}: {job_ops.iloc[0]['Part Number']}")
                            st.write(f"{get_text('num_operations')}: {len(job_ops)}")
                            st.write(f"{get_text('total_processing')}: {job_ops['Run Time'].sum():.1f} {get_text('hours')}")
                            st.write(f"{get_text('total_setup')}: {job_ops['Setup Time'].sum():.1f} {get_text('hours')}")
                            st.write(f"{get_text('total_duration')}: {total_duration:.1f} {get_text('hours')}")
                        else:
                            st.warning(get_text('no_operations'))
                
                # Create and display Gantt chart
                with st.spinner(get_text('generating_schedule')):
                    fig_gantt = create_gantt_chart(df_for_gantt, job_to_highlight)
                    st.plotly_chart(fig_gantt, use_container_width=True)
                    
            except Exception as e:
                st.error(f"{get_text('error_scheduling')}{str(e)}")

    # Add download option if schedule was generated
    if st.session_state.get('schedule_generated', False) and st.session_state.get('complete_scheduled_orders') is not None:
        # Create a copy for export
        export_df = st.session_state['complete_scheduled_orders'].copy()
        
        try:
            # Convert datetime columns to string format
            for col in ['Start Time', 'Finish Time']:
                if not pd.api.types.is_datetime64_any_dtype(export_df[col]):
                    export_df[col] = pd.to_datetime(export_df[col])
                export_df[col] = export_df[col].dt.strftime('%Y-%m-%d %H:%M')
            
            if not pd.api.types.is_datetime64_any_dtype(export_df['Due Date']):
                export_df['Due Date'] = pd.to_datetime(export_df['Due Date'])
            export_df['Due Date'] = export_df['Due Date'].dt.strftime('%Y-%m-%d')
            
            # Create download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                get_text('download_schedule'),
                csv,
                "production_schedule.csv",
                "text/csv",
                key='download-csv'
            )
        except Exception as e:
            st.error(f"{get_text('error_scheduling')}{str(e)}")


if __name__ == "__main__":
    main()
