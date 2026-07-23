const PageHeader = ({ title, subtitle, icon: Icon, actions, backLink }) => {
  return (
    <div className="flex items-start justify-between mb-8 fade-in">
      <div className="flex items-center gap-4">
        {backLink && (
          <a
            href={backLink}
            className="p-2 glass-static rounded-xl hover:scale-105 transition-transform"
            data-testid="back-btn"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
          </a>
        )}
        {Icon && (
          <div className="p-3 rounded-xl" style={{ background: 'rgba(99, 102, 241, 0.15)' }}>
            <Icon className="w-6 h-6" style={{ color: '#818cf8' }} />
          </div>
        )}
        <div>
          <h1 className="text-3xl lg:text-4xl font-bold gradient-text" style={{ fontFamily: 'Space Grotesk' }}>
            {title}
          </h1>
          {subtitle && (
            <p className="text-sm mt-1" style={{ color: 'var(--av-text-muted)' }}>
              {subtitle}
            </p>
          )}
        </div>
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </div>
  );
};

export default PageHeader;
