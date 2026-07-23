import AnimatedCounter from './AnimatedCounter';

const StatWidget = ({ icon, title, value, subtitle, color = 'blue', delay = 0 }) => {
  const colorStyles = {
    blue: {
      iconBg: 'rgba(59, 130, 246, 0.15)',
      iconColor: '#60a5fa',
      border: 'rgba(59, 130, 246, 0.2)',
      glow: '0 0 20px rgba(59, 130, 246, 0.1)',
      gradient: 'linear-gradient(135deg, rgba(59,130,246,0.08) 0%, transparent 60%)',
    },
    violet: {
      iconBg: 'rgba(139, 92, 246, 0.15)',
      iconColor: '#a78bfa',
      border: 'rgba(139, 92, 246, 0.2)',
      glow: '0 0 20px rgba(139, 92, 246, 0.1)',
      gradient: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, transparent 60%)',
    },
    cyan: {
      iconBg: 'rgba(6, 182, 212, 0.15)',
      iconColor: '#22d3ee',
      border: 'rgba(6, 182, 212, 0.2)',
      glow: '0 0 20px rgba(6, 182, 212, 0.1)',
      gradient: 'linear-gradient(135deg, rgba(6,182,212,0.08) 0%, transparent 60%)',
    },
    emerald: {
      iconBg: 'rgba(16, 185, 129, 0.15)',
      iconColor: '#34d399',
      border: 'rgba(16, 185, 129, 0.2)',
      glow: '0 0 20px rgba(16, 185, 129, 0.1)',
      gradient: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, transparent 60%)',
    },
    rose: {
      iconBg: 'rgba(244, 63, 94, 0.15)',
      iconColor: '#fb7185',
      border: 'rgba(244, 63, 94, 0.2)',
      glow: '0 0 20px rgba(244, 63, 94, 0.1)',
      gradient: 'linear-gradient(135deg, rgba(244,63,94,0.08) 0%, transparent 60%)',
    },
    amber: {
      iconBg: 'rgba(245, 158, 11, 0.15)',
      iconColor: '#fbbf24',
      border: 'rgba(245, 158, 11, 0.2)',
      glow: '0 0 20px rgba(245, 158, 11, 0.1)',
      gradient: 'linear-gradient(135deg, rgba(245,158,11,0.08) 0%, transparent 60%)',
    },
  };

  const style = colorStyles[color] || colorStyles.blue;

  return (
    <div
      className="stat-card glass-card"
      style={{
        background: style.gradient,
        borderColor: style.border,
        animationDelay: `${delay}ms`,
      }}
      data-testid={`stat-${title.toLowerCase()}`}
    >
      <div className="flex items-center justify-between mb-4">
        <div
          className="p-3 rounded-xl"
          style={{ background: style.iconBg, color: style.iconColor }}
        >
          {icon}
        </div>
      </div>
      <p className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--av-text-muted)' }}>
        {title}
      </p>
      <p className="text-3xl font-bold" style={{ color: style.iconColor, fontFamily: 'Space Grotesk' }}>
        <AnimatedCounter value={value} />
      </p>
      {subtitle && (
        <p className="text-xs mt-1" style={{ color: 'var(--av-text-muted)' }}>
          {subtitle}
        </p>
      )}
    </div>
  );
};

export default StatWidget;
