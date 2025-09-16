// 主要JavaScript交互逻辑

// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 初始化
    initNavbar();
    initSmoothScroll();
    initScrollAnimations();
    initNetworkVisualization();
    initOpinionLeaderVisualization();
    initPolarizationVisualization();
    initBeliefEvolution();
    initTradingBehavior();
});

// 导航栏功能
function initNavbar() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');
    const navbar = document.querySelector('.navbar');

    // 汉堡菜单切换
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // 点击链接后关闭菜单
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // 滚动时改变导航栏样式
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.backdropFilter = 'blur(10px)';
        } else {
            navbar.style.background = 'white';
            navbar.style.backdropFilter = 'none';
        }

        // 隐藏/显示导航栏
        if (currentScroll > lastScroll && currentScroll > 300) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        lastScroll = currentScroll;
    });
}

// 平滑滚动
function initSmoothScroll() {
    const links = document.querySelectorAll('a[href^="#"]');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetElement.offsetTop - navbarHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// 滚动动画
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');

                // 对于网格项目，添加延迟动画
                if (entry.target.classList.contains('innovation-card') ||
                    entry.target.classList.contains('law-card') ||
                    entry.target.classList.contains('resource-card')) {
                    const cards = entry.target.parentElement.children;
                    Array.from(cards).forEach((card, index) => {
                        setTimeout(() => {
                            card.classList.add('animate-in');
                        }, index * 100);
                    });
                }
            }
        });
    }, observerOptions);

    // 观察需要动画的元素
    const animatedElements = document.querySelectorAll(
        '.innovation-card, .law-card, .resource-card, .arch-layer, .demo-card'
    );

    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// 添加CSS类用于动画
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);

// 社交网络可视化
function initNetworkVisualization() {
    const canvas = document.getElementById('networkCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;

    // 节点和边的数据
    const nodes = [];
    const edges = [];
    const nodeCount = 20;

    // 创建节点
    for (let i = 0; i < nodeCount; i++) {
        nodes.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: Math.random() * 3 + 2,
            isActive: i < 3 // 前3个节点为活跃节点
        });
    }

    // 创建边（随机连接）
    for (let i = 0; i < nodeCount - 1; i++) {
        for (let j = i + 1; j < nodeCount; j++) {
            if (Math.random() > 0.85) { // 15%的概率连接
                edges.push({ from: i, to: j });
            }
        }
    }

    // 动画函数
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 绘制边
        ctx.strokeStyle = 'rgba(30, 58, 138, 0.1)';
        ctx.lineWidth = 1;
        edges.forEach(edge => {
            ctx.beginPath();
            ctx.moveTo(nodes[edge.from].x, nodes[edge.from].y);
            ctx.lineTo(nodes[edge.to].x, nodes[edge.to].y);
            ctx.stroke();
        });

        // 更新和绘制节点
        nodes.forEach(node => {
            // 更新位置
            node.x += node.vx;
            node.y += node.vy;

            // 边界碰撞检测
            if (node.x < node.radius || node.x > canvas.width - node.radius) {
                node.vx *= -1;
            }
            if (node.y < node.radius || node.y > canvas.height - node.radius) {
                node.vy *= -1;
            }

            // 确保节点在画布内
            node.x = Math.max(node.radius, Math.min(canvas.width - node.radius, node.x));
            node.y = Math.max(node.radius, Math.min(canvas.height - node.radius, node.y));

            // 绘制节点
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);

            if (node.isActive) {
                // 活跃节点用橙色
                ctx.fillStyle = '#f59e0b';

                // 添加发光效果
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#f59e0b';
            } else {
                // 普通节点用蓝色
                ctx.fillStyle = '#1e3a8a';
                ctx.shadowBlur = 0;
            }

            ctx.fill();
        });

        // 模拟信息传播波纹
        if (Math.random() > 0.98) {
            const randomNode = nodes[Math.floor(Math.random() * 3)]; // 从活跃节点发出
            createRipple(ctx, randomNode.x, randomNode.y);
        }

        requestAnimationFrame(animate);
    }

    animate();
}

// 创建波纹效果
let ripples = [];

function createRipple(ctx, x, y) {
    ripples.push({
        x: x,
        y: y,
        radius: 0,
        opacity: 0.5
    });

    // 动画波纹
    function animateRipple() {
        ripples = ripples.filter(ripple => {
            ripple.radius += 2;
            ripple.opacity -= 0.01;

            if (ripple.opacity > 0) {
                ctx.strokeStyle = `rgba(245, 158, 11, ${ripple.opacity})`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
                ctx.stroke();
                return true;
            }
            return false;
        });

        if (ripples.length > 0) {
            requestAnimationFrame(animateRipple);
        }
    }

    animateRipple();
}

// 数字递增动画
function animateNumbers() {
    const numbers = document.querySelectorAll('.stat-number');

    numbers.forEach(num => {
        const target = parseInt(num.textContent);
        let current = 0;
        const increment = target / 50;
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            num.textContent = Math.floor(current) + (num.textContent.includes('+') ? '+' : '');
        }, 30);
    });
}

// 当统计数据进入视口时触发动画
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateNumbers();
            statsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
    statsObserver.observe(heroStats);
}

// Opinion Leader Visualization
function initOpinionLeaderVisualization() {
    const canvas = document.getElementById('opinionLeaderCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 300;

    // 创建更复杂的网络结构
    const nodes = [];
    const edges = [];

    // 创建意见领袖（中心的几个大节点）
    const leaders = [
        { x: 200, y: 150, radius: 12, influence: 1, isLeader: true, id: 0 },
        { x: 120, y: 100, radius: 10, influence: 0.8, isLeader: true, id: 1 },
        { x: 280, y: 110, radius: 10, influence: 0.8, isLeader: true, id: 2 }
    ];

    leaders.forEach(leader => nodes.push(leader));

    // 创建多层次的普通节点
    let nodeId = 3;

    // 第一层：直接连接到领袖的节点
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 5; j++) {
            const angle = (Math.PI * 2 * j) / 5 + (i * 0.5);
            const distance = 40 + Math.random() * 20;
            const leader = leaders[i];
            const node = {
                x: leader.x + Math.cos(angle) * distance,
                y: leader.y + Math.sin(angle) * distance,
                radius: 4,
                influence: 0,
                isLeader: false,
                id: nodeId++,
                layer: 1,
                parentId: i
            };
            nodes.push(node);
            edges.push({ from: i, to: node.id });
        }
    }

    // 第二层：外围节点
    for (let i = 0; i < 30; i++) {
        const angle = (Math.PI * 2 * i) / 30;
        const distance = 100 + Math.random() * 50;
        const node = {
            x: 200 + Math.cos(angle) * distance,
            y: 150 + Math.sin(angle) * distance,
            radius: 3,
            influence: 0,
            isLeader: false,
            id: nodeId++,
            layer: 2
        };
        nodes.push(node);

        // 连接到最近的第一层节点
        let minDist = Infinity;
        let closestNode = null;
        nodes.forEach(n => {
            if (n.layer === 1) {
                const dist = Math.sqrt(Math.pow(n.x - node.x, 2) + Math.pow(n.y - node.y, 2));
                if (dist < minDist) {
                    minDist = dist;
                    closestNode = n;
                }
            }
        });
        if (closestNode && minDist < 60) {
            edges.push({ from: closestNode.id, to: node.id });
        }
    }

    // 添加一些横向连接
    nodes.forEach(node => {
        if (!node.isLeader) {
            nodes.forEach(other => {
                if (!other.isLeader && node.id < other.id) {
                    const dist = Math.sqrt(Math.pow(node.x - other.x, 2) + Math.pow(node.y - other.y, 2));
                    if (dist < 30 && Math.random() > 0.7) {
                        edges.push({ from: node.id, to: other.id });
                    }
                }
            });
        }
    });

    // 信息传播状态
    let propagationTime = 0;

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        propagationTime += 0.02;

        // 绘制边
        edges.forEach(edge => {
            const fromNode = nodes.find(n => n.id === edge.from);
            const toNode = nodes.find(n => n.id === edge.to);

            if (fromNode && toNode) {
                // 根据影响力决定边的可见度
                const opacity = Math.min(fromNode.influence, toNode.influence) * 0.3 + 0.05;
                ctx.strokeStyle = `rgba(30, 58, 138, ${opacity})`;
                ctx.lineWidth = 0.5 + Math.min(fromNode.influence, toNode.influence);
                ctx.beginPath();
                ctx.moveTo(fromNode.x, fromNode.y);
                ctx.lineTo(toNode.x, toNode.y);
                ctx.stroke();
            }
        });

        // 更新和绘制节点
        nodes.forEach(node => {
            if (!node.isLeader) {
                // 计算从领袖传播过来的影响
                let maxInfluence = 0;
                edges.forEach(edge => {
                    if (edge.to === node.id) {
                        const fromNode = nodes.find(n => n.id === edge.from);
                        if (fromNode) {
                            const delay = node.layer * 0.5;
                            const influence = fromNode.influence * Math.max(0, Math.min(1, (propagationTime - delay) * 0.3));
                            maxInfluence = Math.max(maxInfluence, influence);
                        }
                    }
                });
                node.influence = maxInfluence;
            }

            // 绘制节点
            if (node.isLeader) {
                // 领袖节点 - 橙色，带脉动效果
                const pulse = Math.sin(propagationTime * 3) * 0.2 + 1;
                ctx.fillStyle = '#f59e0b';
                ctx.shadowBlur = 15 * pulse;
                ctx.shadowColor = '#f59e0b';
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius * pulse, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 0;
            } else {
                // 普通节点 - 根据影响力改变颜色
                const r = 30 + (215 * node.influence);
                const g = 58 + (100 * node.influence);
                const b = 138 - (127 * node.influence);
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius + node.influence * 2, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        // 绘制影响力波纹
        leaders.forEach(leader => {
            const waveRadius = (propagationTime * 30) % 150;
            const waveOpacity = Math.max(0, 1 - waveRadius / 150);
            ctx.strokeStyle = `rgba(245, 158, 11, ${waveOpacity * 0.3})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(leader.x, leader.y, waveRadius, 0, Math.PI * 2);
            ctx.stroke();
        });

        requestAnimationFrame(animate);
    }

    animate();
}

// Polarization Visualization
function initPolarizationVisualization() {
    const canvas = document.getElementById('polarizationCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 300;

    // 两组节点
    const group1 = [];
    const group2 = [];

    // 创建第一组（左侧）
    for (let i = 0; i < 15; i++) {
        group1.push({
            x: 100 + Math.random() * 60 - 30,
            y: 150 + Math.random() * 100 - 50,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: 4 + Math.random() * 2,
            color: '#3b82f6',
            belief: -1
        });
    }

    // 创建第二组（右侧）
    for (let i = 0; i < 15; i++) {
        group2.push({
            x: 300 + Math.random() * 60 - 30,
            y: 150 + Math.random() * 100 - 50,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: 4 + Math.random() * 2,
            color: '#ef4444',
            belief: 1
        });
    }

    const allNodes = [...group1, ...group2];

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 绘制分隔线
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(200, 50);
        ctx.lineTo(200, 250);
        ctx.stroke();
        ctx.setLineDash([]);

        // 更新和绘制节点
        allNodes.forEach(node => {
            // 更新位置
            node.x += node.vx;
            node.y += node.vy;

            // 群体内聚力
            const centerX = node.belief < 0 ? 100 : 300;
            const centerY = 150;
            const dx = centerX - node.x;
            const dy = centerY - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist > 60) {
                node.vx += dx * 0.001;
                node.vy += dy * 0.001;
            }

            // 边界反弹
            if (node.x < node.radius || node.x > canvas.width - node.radius) {
                node.vx *= -0.8;
            }
            if (node.y < node.radius || node.y > canvas.height - node.radius) {
                node.vy *= -0.8;
            }

            // 速度衰减
            node.vx *= 0.98;
            node.vy *= 0.98;

            // 绘制节点
            ctx.fillStyle = node.color;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fill();

            // 绘制连线（同组内）
            allNodes.forEach(other => {
                if (other !== node && Math.sign(other.belief) === Math.sign(node.belief)) {
                    const distance = Math.sqrt(Math.pow(other.x - node.x, 2) + Math.pow(other.y - node.y, 2));
                    if (distance < 40) {
                        ctx.strokeStyle = node.color;
                        ctx.globalAlpha = 0.1;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(node.x, node.y);
                        ctx.lineTo(other.x, other.y);
                        ctx.stroke();
                    }
                }
            });
            ctx.globalAlpha = 1;
        });

        // 添加标签
        ctx.fillStyle = '#3b82f6';
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Group A', 100, 30);

        ctx.fillStyle = '#ef4444';
        ctx.fillText('Group B', 300, 30);

        requestAnimationFrame(animate);
    }

    animate();
}

// Belief Evolution Visualization
function initBeliefEvolution() {
    const canvas = document.getElementById('beliefEvolutionCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 350;
    canvas.height = 250;

    // 初始化粒子
    const particles = [];
    const particleCount = 80;

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: 175 + (Math.random() - 0.5) * 50,
            y: 125 + (Math.random() - 0.5) * 30,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            belief: 0, // -1 to 1
            targetBelief: 0,
            color: '#94a3b8',
            radius: 2,
            group: Math.random() > 0.5 ? 1 : -1
        });
    }

    let rumorTime = 0;
    let rumorActive = false;

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 背景渐变
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.05)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0)');
        gradient.addColorStop(1, 'rgba(239, 68, 68, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        rumorTime += 0.015;

        // 周期性激活谣言
        if (rumorTime > 2) {
            rumorActive = true;
        }
        if (rumorTime > 8) {
            rumorTime = 0;
            rumorActive = false;
            // 重置粒子
            particles.forEach(p => {
                p.targetBelief = 0;
                p.group = Math.random() > 0.5 ? 1 : -1;
            });
        }

        // 更新粒子
        particles.forEach(particle => {
            if (rumorActive) {
                // 谣言传播时，粒子开始极化
                particle.targetBelief = particle.group * 0.8;

                // 向两侧移动
                const targetX = particle.group > 0 ? 280 : 70;
                particle.vx += (targetX - particle.x) * 0.001;
            } else {
                // 正常状态，向中心聚集
                particle.vx += (175 - particle.x) * 0.0005;
                particle.targetBelief = 0;
            }

            // 更新信念值
            particle.belief += (particle.targetBelief - particle.belief) * 0.05;

            // 更新位置
            particle.x += particle.vx;
            particle.y += particle.vy;

            // 边界反弹
            if (particle.x < 5 || particle.x > canvas.width - 5) {
                particle.vx *= -0.5;
            }
            if (particle.y < 5 || particle.y > canvas.height - 5) {
                particle.vy *= -0.5;
            }

            // 速度衰减
            particle.vx *= 0.98;
            particle.vy *= 0.98;

            // 随机运动
            particle.vy += (Math.random() - 0.5) * 0.1;

            // 根据信念值设置颜色
            if (particle.belief > 0.1) {
                const intensity = Math.min(1, Math.abs(particle.belief));
                particle.color = `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`;
            } else if (particle.belief < -0.1) {
                const intensity = Math.min(1, Math.abs(particle.belief));
                particle.color = `rgba(59, 130, 246, ${0.3 + intensity * 0.7})`;
            } else {
                particle.color = 'rgba(148, 163, 184, 0.8)';
            }

            // 绘制粒子
            ctx.fillStyle = particle.color;
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.radius + Math.abs(particle.belief) * 2, 0, Math.PI * 2);
            ctx.fill();
        });

        // 绘制中线
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(175, 20);
        ctx.lineTo(175, canvas.height - 20);
        ctx.stroke();
        ctx.setLineDash([]);

        // 添加标签
        ctx.fillStyle = '#1e293b';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';

        if (!rumorActive) {
            ctx.fillText('Neutral State', 175, canvas.height - 10);
        } else {
            ctx.fillStyle = '#3b82f6';
            ctx.fillText('Optimistic', 70, canvas.height - 10);
            ctx.fillStyle = '#ef4444';
            ctx.fillText('Pessimistic', 280, canvas.height - 10);
        }

        // 显示谣言状态
        if (rumorActive) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
            ctx.font = 'bold 12px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('RUMOR SPREADING', 175, 20);
        }

        requestAnimationFrame(animate);
    }

    animate();
}

// Trading Behavior Visualization
function initTradingBehavior() {
    const canvas = document.getElementById('tradingBehaviorCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 350;
    canvas.height = 250;

    // 交易行为数据
    const traders = [];
    for (let i = 0; i < 40; i++) {
        traders.push({
            x: Math.random() * canvas.width,
            y: 125,
            action: 'hold', // buy, sell, hold
            intensity: 0,
            targetY: 125,
            vx: (Math.random() - 0.5) * 2,
            vy: 0,
            radius: 3 + Math.random() * 2
        });
    }

    let marketTime = 0;
    let volatilityPhase = false;

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        marketTime += 0.02;

        // 周期性切换市场状态
        volatilityPhase = Math.sin(marketTime) > 0;

        // 绘制背景区域
        ctx.fillStyle = 'rgba(34, 197, 94, 0.1)';
        ctx.fillRect(0, 0, canvas.width, 85);
        ctx.fillStyle = 'rgba(148, 163, 184, 0.05)';
        ctx.fillRect(0, 85, canvas.width, 80);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
        ctx.fillRect(0, 165, canvas.width, 85);

        // 绘制分割线
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, 85);
        ctx.lineTo(canvas.width, 85);
        ctx.moveTo(0, 165);
        ctx.lineTo(canvas.width, 165);
        ctx.stroke();

        // 更新交易者
        traders.forEach(trader => {
            if (volatilityPhase) {
                // 高波动期 - 极化交易行为
                const decision = Math.random();
                if (decision < 0.4) {
                    trader.action = 'sell';
                    trader.targetY = 210;
                    trader.intensity = 0.8;
                } else if (decision < 0.6) {
                    trader.action = 'buy';
                    trader.targetY = 40;
                    trader.intensity = 0.8;
                } else {
                    trader.action = 'hold';
                    trader.targetY = 125;
                    trader.intensity = 0.3;
                }
            } else {
                // 正常期 - 均衡交易
                trader.action = 'hold';
                trader.targetY = 125;
                trader.intensity = 0.2;
            }

            // 更新位置
            trader.vy += (trader.targetY - trader.y) * 0.02;
            trader.y += trader.vy;
            trader.x += trader.vx;

            // 边界处理
            if (trader.x < 0) trader.x = canvas.width;
            if (trader.x > canvas.width) trader.x = 0;

            // 速度衰减
            trader.vy *= 0.9;

            // 绘制交易者
            let color;
            if (trader.action === 'buy') {
                color = `rgba(34, 197, 94, ${0.3 + trader.intensity * 0.7})`;
            } else if (trader.action === 'sell') {
                color = `rgba(239, 68, 68, ${0.3 + trader.intensity * 0.7})`;
            } else {
                color = `rgba(148, 163, 184, ${0.3 + trader.intensity * 0.7})`;
            }

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(trader.x, trader.y, trader.radius + trader.intensity * 2, 0, Math.PI * 2);
            ctx.fill();

            // 绘制交易线
            if (trader.action !== 'hold' && trader.intensity > 0.5) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 0.5;
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                ctx.moveTo(trader.x, trader.y);
                ctx.lineTo(trader.x, 125);
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
        });

        // 添加标签
        ctx.fillStyle = '#22c55e';
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'left';
        ctx.fillText('BUY', 10, 50);

        ctx.fillStyle = '#94a3b8';
        ctx.fillText('HOLD', 10, 130);

        ctx.fillStyle = '#ef4444';
        ctx.fillText('SELL', 10, 210);

        // 市场状态指示器
        ctx.textAlign = 'center';
        if (volatilityPhase) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
            ctx.font = 'bold 12px Inter';
            ctx.fillText('HIGH VOLATILITY', 175, 20);
        } else {
            ctx.fillStyle = 'rgba(34, 197, 94, 0.8)';
            ctx.font = 'bold 12px Inter';
            ctx.fillText('STABLE MARKET', 175, 20);
        }

        requestAnimationFrame(animate);
    }

    animate();
}

// 添加页面加载动画
window.addEventListener('load', () => {
    document.body.classList.add('loaded');
});