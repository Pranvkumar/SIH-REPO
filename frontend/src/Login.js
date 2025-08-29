import React, { useState } from 'react';

const Login = ({ onLogin, onNavigate }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });
  const [userType, setUserType] = useState('citizen');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Demo credentials
  const demoCredentials = {
    citizen: { username: 'citizen', password: 'ocean123' },
    admin: { username: 'admin', password: 'admin123' },
    authority: { username: 'authority', password: 'auth456' }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Simulate API call delay
    setTimeout(() => {
      const demo = demoCredentials[userType];
      
      if (credentials.username === demo.username && credentials.password === demo.password) {
        const userData = {
          id: Date.now(),
          username: credentials.username,
          type: userType,
          name: userType === 'citizen' ? 'John Citizen' : 
                userType === 'admin' ? 'Admin User' : 'Authority Officer',
          permissions: userType === 'citizen' ? ['report'] : 
                      userType === 'admin' ? ['admin', 'reports', 'dashboard'] : 
                      ['dashboard', 'reports', 'manage']
        };
        
        localStorage.setItem('oceanUser', JSON.stringify(userData));
        onLogin(userData);
      } else {
        setError('Invalid credentials. Please check username and password.');
      }
      setIsLoading(false);
    }, 1500);
  };

  const handleDemoLogin = (type) => {
    const demo = demoCredentials[type];
    setCredentials({ username: demo.username, password: demo.password });
    setUserType(type);
  };

  const getUserTypeDescription = () => {
    switch (userType) {
      case 'citizen':
        return 'Report ocean hazards and view public dashboard';
      case 'admin':
        return 'Full system access with administrative privileges';
      case 'authority':
        return 'Monitor hazards and manage emergency responses';
      default:
        return '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 flex items-center justify-center py-8">
      <div className="absolute inset-0 bg-black bg-opacity-20"></div>
      
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-blue-400 bg-opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-96 h-96 rounded-full bg-cyan-400 bg-opacity-20 animate-pulse" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full bg-purple-400 bg-opacity-10 animate-pulse" style={{ animationDelay: '4s' }}></div>
      </div>

      <div className="relative z-10 w-full max-w-md px-6">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full mx-auto mb-4 flex items-center justify-center shadow-2xl">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Ocean Hazard Alert</h1>
          <p className="text-blue-200">Secure Login Portal</p>
        </div>

        {/* Login Card */}
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl border border-white border-opacity-20 p-8 shadow-2xl">
          {/* User Type Selector */}
          <div className="mb-6">
            <label className="block text-sm font-semibold text-white mb-3">Login As</label>
            <div className="grid grid-cols-3 gap-2">
              {['citizen', 'admin', 'authority'].map((type) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => setUserType(type)}
                  className={`py-2 px-3 text-xs font-medium rounded-lg transition-all duration-300 ${
                    userType === type
                      ? 'bg-blue-500 text-white shadow-lg'
                      : 'bg-white bg-opacity-20 text-blue-200 hover:bg-opacity-30'
                  }`}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
            <p className="text-xs text-blue-300 mt-2">{getUserTypeDescription()}</p>
          </div>

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Username Field */}
            <div>
              <label className="block text-sm font-semibold text-white mb-2">Username</label>
              <div className="relative">
                <input
                  type="text"
                  value={credentials.username}
                  onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
                  className="w-full px-4 py-3 bg-white bg-opacity-20 border border-white border-opacity-30 rounded-lg text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                  placeholder="Enter your username"
                  required
                />
                <div className="absolute inset-y-0 right-3 flex items-center">
                  <svg className="w-5 h-5 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label className="block text-sm font-semibold text-white mb-2">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={credentials.password}
                  onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                  className="w-full px-4 py-3 bg-white bg-opacity-20 border border-white border-opacity-30 rounded-lg text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-3 flex items-center text-blue-300 hover:text-white transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={showPassword ? "M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21" : "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"} />
                  </svg>
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-500 bg-opacity-20 border border-red-500 border-opacity-50 rounded-lg p-3">
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}

            {/* Login Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-3 px-6 rounded-lg transition-all duration-300 transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Authenticating...
                </div>
              ) : (
                'Sign In'
              )}
            </button>

            {/* Demo Login Buttons */}
            <div className="mt-6 pt-6 border-t border-white border-opacity-20">
              <p className="text-sm text-blue-300 text-center mb-4">Quick Demo Access</p>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => handleDemoLogin('citizen')}
                  className="py-2 px-3 bg-green-600 bg-opacity-80 hover:bg-opacity-100 text-white text-xs rounded transition-all duration-300"
                >
                  Citizen Demo
                </button>
                <button
                  type="button"
                  onClick={() => handleDemoLogin('admin')}
                  className="py-2 px-3 bg-red-600 bg-opacity-80 hover:bg-opacity-100 text-white text-xs rounded transition-all duration-300"
                >
                  Admin Demo
                </button>
                <button
                  type="button"
                  onClick={() => handleDemoLogin('authority')}
                  className="py-2 px-3 bg-purple-600 bg-opacity-80 hover:bg-opacity-100 text-white text-xs rounded transition-all duration-300"
                >
                  Authority Demo
                </button>
              </div>
            </div>
          </form>

          {/* Footer */}
          <div className="mt-8 text-center">
            <button
              onClick={() => onNavigate('home')}
              className="text-blue-300 hover:text-white text-sm transition-colors duration-300"
            >
              Continue without login →
            </button>
          </div>
        </div>

        {/* Demo Credentials Info */}
        <div className="mt-6 bg-white bg-opacity-5 backdrop-blur-sm rounded-lg p-4 border border-white border-opacity-10">
          <h3 className="text-white font-semibold text-sm mb-2">Demo Credentials:</h3>
          <div className="space-y-1 text-xs text-blue-200">
            <p><span className="font-medium">Citizen:</span> citizen / ocean123</p>
            <p><span className="font-medium">Admin:</span> admin / admin123</p>
            <p><span className="font-medium">Authority:</span> authority / auth456</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;