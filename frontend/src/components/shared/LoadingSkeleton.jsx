const LoadingSkeleton = ({ rows = 5, type = 'table' }) => {
  if (type === 'cards') {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="glass-static rounded-2xl p-6 space-y-4" style={{ animationDelay: `${i * 100}ms` }}>
            <div className="skeleton h-10 w-10 rounded-xl" />
            <div className="skeleton h-4 w-20" />
            <div className="skeleton h-8 w-24" />
          </div>
        ))}
      </div>
    );
  }

  if (type === 'chart') {
    return (
      <div className="glass-static rounded-2xl p-6 space-y-4">
        <div className="skeleton h-6 w-40" />
        <div className="skeleton h-64 w-full rounded-xl" />
      </div>
    );
  }

  return (
    <div className="glass-static rounded-2xl p-6 space-y-3">
      <div className="skeleton h-6 w-48 mb-6" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4" style={{ animationDelay: `${i * 50}ms` }}>
          <div className="skeleton h-5 w-8" />
          <div className="skeleton h-5 flex-1" />
          <div className="skeleton h-5 w-24" />
          <div className="skeleton h-5 w-16" />
        </div>
      ))}
    </div>
  );
};

export default LoadingSkeleton;
