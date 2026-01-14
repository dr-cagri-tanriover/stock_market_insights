"""
Printing utilities for dataframes using Rich library.

This module provides functions to beautifully display pandas DataFrames
in the terminal using Rich's formatting capabilities.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import Optional


def print_dataframe(
    df: pd.DataFrame,
    title: Optional[str] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    show_index: bool = True,
    justify_numeric: str = "right"
) -> None:
    """
    Print a pandas DataFrame to the terminal using Rich library.
    
    Args:
        df: The pandas DataFrame to print
        title: Optional title for the table
        max_rows: Maximum number of rows to display (None for all rows)
        max_cols: Maximum number of columns to display (None for all columns)
        show_index: Whether to show the index column
        justify_numeric: Justification for numeric columns. Options: "left", "center", "right" (default: "right")
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> print_dataframe(df, title="My DataFrame")
        >>> print_dataframe(df, title="My DataFrame", justify_numeric="center")
    """
    console = Console()
    
    # Validate justify_numeric parameter
    valid_justifications = ["left", "center", "right"]
    if justify_numeric not in valid_justifications:
        raise ValueError(f"justify_numeric must be one of {valid_justifications}, got '{justify_numeric}'")
    
    # Handle empty dataframe
    if df.empty:
        console.print("[yellow]DataFrame is empty[/yellow]")
        return
    
    # Limit rows if specified
    display_df = df.head(max_rows) if max_rows else df
    
    # Limit columns if specified
    if max_cols:
        display_df = display_df.iloc[:, :max_cols]
    
    # Create Rich table
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add title if provided
    if title:
        table.title = title
    
    # Add index column if requested
    if show_index:
        table.add_column("Index", style="dim", justify="right")
    
    # Add data columns
    for col in display_df.columns:
        # Determine column alignment based on data type
        if pd.api.types.is_numeric_dtype(display_df[col]):
            justify = justify_numeric
        else:
            justify = "left"
        
        table.add_column(str(col), justify=justify, overflow="fold")
    
    # Add rows
    for idx, row in display_df.iterrows():
        row_data = []
        
        # Add index if requested
        if show_index:
            row_data.append(str(idx))
        
        # Add row values
        for col in display_df.columns:
            value = row[col]
            
            # Format the value based on type
            if pd.isna(value):
                formatted_value = "[dim]NaN[/dim]"
            elif pd.api.types.is_numeric_dtype(display_df[col]):
                # Format numbers nicely
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2e}"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            row_data.append(formatted_value)
        
        table.add_row(*row_data)
    
    # Add summary information if rows/cols were limited
    if max_rows and len(df) > max_rows:
        table.caption = f"Showing {max_rows} of {len(df)} rows"
    elif max_cols and len(df.columns) > max_cols:
        table.caption = f"Showing {max_cols} of {len(df.columns)} columns"
    elif (max_rows and len(df) > max_rows) and (max_cols and len(df.columns) > max_cols):
        table.caption = f"Showing {max_rows} of {len(df)} rows, {max_cols} of {len(df.columns)} columns"
    
    # Print the table
    console.print(table)
    
    # Print shape information
    if max_rows or max_cols:
        console.print(f"\n[dim]DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns[/dim]")


def print_dataframe_simple(df: pd.DataFrame, title: Optional[str] = None) -> None:
    """
    Simple wrapper to print dataframe using Rich's console.
    This is a simpler alternative that uses Rich's built-in dataframe rendering.
    
    Args:
        df: The pandas DataFrame to print
        title: Optional title for the output
    """
    console = Console()
    
    if title:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
    
    console.print(df)

