// 图表可视化配置和初始化

// 图表配置
const chartColors = {
    primary: '#1e3a8a',
    secondary: '#f59e0b',
    success: '#10b981',
    danger: '#ef4444',
    info: '#3b82f6',
    warning: '#f59e0b',
    light: '#f3f4f6',
    dark: '#1f2937'
};

// 初始化所有图表
document.addEventListener('DOMContentLoaded', function() {
    initAblationChart();
    initScalabilityChart();
});

// 消融实验图表
function initAblationChart() {
    const ctx = document.getElementById('ablationChart');
    if (!ctx) return;

    const ablationData = {
        labels: ['完整模型', '无社交网络', '无新闻分析', '无技术指标', '无个性化', '随机决策'],
        datasets: [
            {
                label: '动量效应',
                data: [0.92, 0.73, 0.81, 0.78, 0.65, 0.12],
                backgroundColor: chartColors.primary,
                borderColor: chartColors.primary,
                borderWidth: 2,
                tension: 0.4
            },
            {
                label: '羊群效应',
                data: [0.88, 0.52, 0.76, 0.72, 0.61, 0.08],
                backgroundColor: chartColors.secondary,
                borderColor: chartColors.secondary,
                borderWidth: 2,
                tension: 0.4
            },
            {
                label: '处置效应',
                data: [0.85, 0.71, 0.73, 0.69, 0.58, 0.15],
                backgroundColor: chartColors.success,
                borderColor: chartColors.success,
                borderWidth: 2,
                tension: 0.4
            },
            {
                label: '波动聚集',
                data: [0.90, 0.78, 0.82, 0.75, 0.63, 0.21],
                backgroundColor: chartColors.info,
                borderColor: chartColors.info,
                borderWidth: 2,
                tension: 0.4
            }
        ]
    };

    new Chart(ctx, {
        type: 'radar',
        data: ablationData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                title: {
                    display: true,
                    text: '不同组件对市场定律验证的贡献',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.r * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    pointLabels: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
}

// 可扩展性测试图表
function initScalabilityChart() {
    const ctx = document.getElementById('scalabilityChart');
    if (!ctx) return;

    const scalabilityData = {
        labels: ['100智能体', '500智能体', '1000智能体'],
        datasets: [
            {
                label: '市场定律验证率',
                data: [83, 89, 92],
                backgroundColor: chartColors.primary + '80',
                borderColor: chartColors.primary,
                borderWidth: 3,
                yAxisID: 'y',
                type: 'line',
                tension: 0.4,
                fill: true
            },
            {
                label: '计算时间 (小时)',
                data: [2.5, 8.3, 18.6],
                backgroundColor: chartColors.secondary + '60',
                borderColor: chartColors.secondary,
                borderWidth: 2,
                yAxisID: 'y1',
                type: 'bar'
            },
            {
                label: 'API调用次数 (万次)',
                data: [5.2, 24.8, 51.3],
                backgroundColor: chartColors.success + '60',
                borderColor: chartColors.success,
                borderWidth: 2,
                yAxisID: 'y1',
                type: 'bar'
            }
        ]
    };

    new Chart(ctx, {
        type: 'line',
        data: scalabilityData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                title: {
                    display: true,
                    text: '系统可扩展性与性能表现',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label + ': ';
                            if (context.parsed.y !== null) {
                                if (context.datasetIndex === 0) {
                                    label += context.parsed.y + '%';
                                } else if (context.datasetIndex === 1) {
                                    label += context.parsed.y + ' 小时';
                                } else {
                                    label += context.parsed.y + ' 万次';
                                }
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '验证率 (%)',
                        font: {
                            size: 12
                        }
                    },
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '资源消耗',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// 添加动态数据更新演示（可选）
function updateChartsWithAnimation() {
    // 可以在这里添加动态更新图表数据的功能
    // 例如：模拟实时数据更新
}

// 响应窗口大小变化
window.addEventListener('resize', function() {
    // Chart.js会自动处理响应式，但可以在这里添加额外的逻辑
});

// 为图表添加下载功能
function downloadChart(chartId) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;

    const url = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = chartId + '.png';
    link.href = url;
    link.click();
}

// 导出函数供外部使用
window.chartFunctions = {
    downloadChart: downloadChart,
    updateChartsWithAnimation: updateChartsWithAnimation
};