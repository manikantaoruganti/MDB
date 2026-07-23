import { useState, useEffect, useCallback } from 'react';

/**
 * Generic async data-fetching hook with loading/error states.
 * @param {Function} fetchFn  – async function that returns data
 * @param {Array}    deps     – dependency array (re-fetches when deps change)
 * @param {object}   options  – { immediate: bool }
 */
const useApi = (fetchFn, deps = [], options = { immediate: true }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(async (...args) => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchFn(...args);
      setData(result);
      return result;
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'An error occurred');
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchFn]);

  useEffect(() => {
    if (options.immediate) {
      execute();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error, execute, setData };
};

export default useApi;
