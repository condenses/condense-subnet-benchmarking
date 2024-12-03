def plot_fancy(condense_acc, causal_acc, condense_tokens, causal_tokens, condense_std, causal_std):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Accuracy Comparison', 'Token Usage Analysis'),
        horizontal_spacing=0.1
    )

    # Colors
    colors = ['#636EFA', '#EF553B']

    # Add accuracy bars
    fig.add_trace(
        go.Bar(
            x=['Condense', 'Causal'],
            y=[condense_acc, causal_acc],
            name='Accuracy',
            marker_color=colors,
            text=[f'{condense_acc:.2%}', f'{causal_acc:.2%}'],
            textposition='auto',
        ),
        row=1, col=1
    )

    # Add token usage bars with error bars
    fig.add_trace(
        go.Bar(
            x=['Condense', 'Causal'],
            y=[condense_tokens, causal_tokens],
            name='Tokens',
            marker_color=colors,
            error_y=dict(
                type='data',
                array=[condense_std, causal_std],
                visible=True,
                color='rgba(0,0,0,0.5)'
            ),
            text=[f'{condense_tokens:.0f}±{condense_std:.0f}', 
                  f'{causal_tokens:.0f}±{causal_std:.0f}'],
            textposition='auto',
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        title={
            'text': 'Model Performance Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        height=600,
        width=1200,
    )

    # Update y-axes
    fig.update_yaxes(title_text='Accuracy', range=[0, 1], tickformat='.0%', row=1, col=1)
    fig.update_yaxes(title_text='Number of Tokens', row=1, col=2)

    # Save the plot
    fig.write_html('results/fancy_comparison_plots.html')
    fig.write_image('results/fancy_comparison_plots.png')

def plot_condense_time_series(dates: list[str], accuracies: list[float], token_counts: list[int], original_accuracy: float, original_mean_token_count: int, title: str):
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    # Define size mapping function
    def get_marker_size(tokens):
        if tokens < 1024:
            return 8  # small
        elif tokens <= 4096:
            return 15  # medium
        else:
            return 25  # large

    # Convert token counts to marker sizes
    marker_sizes = [get_marker_size(tokens) for tokens in token_counts]

    fig = go.Figure()

    # Calculate x-axis range with padding
    date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    if len(dates) == 1:
        x_min = (date_objects[0] - timedelta(days=5)).strftime('%Y-%m-%d')
        x_max = (date_objects[0] + timedelta(days=5)).strftime('%Y-%m-%d')
    else:
        x_min = dates[0]
        x_max = dates[-1]

    # Add horizontal line for original accuracy
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],  # span entire x-axis with padding
            y=[original_accuracy, original_accuracy],
            mode='lines',
            name='Original Accuracy',
            line=dict(
                color='#EF553B',
                width=2,
                dash='dash'
            ),
            hovertemplate='Original Accuracy: %{y:.1%}<extra></extra>'
        )
    )

    # Add scatter plot for condense accuracy
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=accuracies,
            mode='lines+markers',
            name='Condense Accuracy',
            line=dict(color='#636EFA', width=2),
            marker=dict(
                size=marker_sizes,
                color='#636EFA',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            hovertemplate=(
                '<b>Date:</b> %{x}<br>' +
                '<b>Accuracy:</b> %{y:.1%}<br>' +
                '<b>Tokens:</b> %{customdata:,}<extra></extra>'
            ),
            customdata=token_counts
        )
    )
    size_legend = [(512, 8), (2048, 15), (8192, 25)]
    # Add legend for marker sizes
    for tokens, size in size_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=size, color='#636EFA'),
                name=f'{"<" if tokens==1024 else ">"if tokens==4096 else ""}{tokens} tokens',
                legendgroup='token_sizes',
                legendgrouptitle_text='Input Token'
            )
        )

    # Add legend for original token count
    # Set original_size 
    original_size = size_legend[0]
    for tokens, size in size_legend:
        if tokens < original_mean_token_count:
            original_size = size
            print(original_size)
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode='markers', name=f'Original Input Token: {original_mean_token_count:,.0f} tokens', legendgroup='original', line=dict(color='#EF553B', width=2, dash='dash'), marker=dict(size=original_size))
    )


    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        template='plotly_white',
        hovermode='x unified',
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            tracegroupgap=5
        ),
        yaxis=dict(
            title='Accuracy',
            tickformat='.0%',
            range=[0, 1]  # Fixed range from 0 to 100%
        ),
        xaxis=dict(
            title='Date'
        )
    )

    # Save the plot
    fig.write_html(f'results/{title}.html')
    fig.write_image(f'results/{title}.png')

if __name__ == "__main__":
    dates = ['2024-12-03']
    accuracies=[0.35]
    token_counts=[931]
    original_token = 3365
    original_accuracy = 0.85
    title = "Condense Subnet Benchmark: RULER-4K-qa"

    plot_condense_time_series(
        dates,
        accuracies,
        token_counts,
        original_accuracy,
        original_token,
        title
    )