"""SQL grammar templates for supported analytical-spec shapes.

Only SQL structure appears here. Names, expressions, predicates, limits, and
sort directions are validated by ``SQLGenerator`` before template rendering.
"""

TEMPLATES = {
    "aggregate": """
        SELECT {agg_expr} AS value
        FROM {view}
        {where}
    """,
    "group_by": """
        SELECT {dimension}, {agg_expr} AS value
        FROM {view}
        {where}
        GROUP BY {dimension}
        ORDER BY value DESC
    """,
    "rank": """
        SELECT {dimension}, {agg_expr} AS value
        FROM {view}
        {where}
        GROUP BY {dimension}
        ORDER BY value {direction}
        LIMIT {limit}
    """,
    "rank_partitioned": """
        WITH ranked AS (
            SELECT {group_dims},
                   {agg_expr} AS value,
                   DENSE_RANK() OVER (
                       PARTITION BY {partition_dims}
                       ORDER BY {agg_expr} {direction}
                   ) AS rnk
            FROM {view}
            {where}
            GROUP BY {group_dims}
        )
        SELECT * FROM ranked WHERE rnk <= {limit}
    """,
    "contribution": """
        SELECT {dimension},
               {agg_expr} AS value,
               ROUND(100.0 * {agg_expr}
                     / SUM({agg_expr}) OVER (), 2) AS pct
        FROM {view}
        {where}
        GROUP BY {dimension}
        ORDER BY value DESC
    """,
    "compare": """
        SELECT s.{dimension},
               {agg_expr} AS actual,
               {target_expr} AS target
        FROM {view} s
        JOIN {targets_table} t
          ON s.{dimension} = t.{dimension}
         AND s.{time_column} = t.{time_column}
        {where}
        GROUP BY s.{dimension}, {target_expr}
        HAVING {agg_expr} {mf_operator} {target_expr}
    """,
    "trend": """
        WITH yearly AS (
            SELECT year, {agg_expr} AS val
            FROM {view}
            {where}
            GROUP BY year
        )
        SELECT year, val,
               ROUND(
                   (val - LAG(val) OVER (ORDER BY year))
                   / NULLIF(LAG(val) OVER (ORDER BY year), 0),
                   4
               ) AS yoy_growth
        FROM yearly
        ORDER BY year
    """,
}
