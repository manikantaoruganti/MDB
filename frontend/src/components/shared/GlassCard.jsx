const GlassCard = ({ children, className = '', hover = true, glow = false, padding = true, ...props }) => {
  return (
    <div
      className={`${hover ? 'glass-card' : 'glass-static rounded-[1.25rem]'} ${padding ? 'p-6' : ''} ${glow ? 'glow-blue' : ''} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export default GlassCard;
