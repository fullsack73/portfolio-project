import React from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';

const Selector = ({ activeView, onViewChange, isOpen, onToggle }) => {
    const { t } = useTranslation();

    return (
        <>
            <button 
                className={`menu-toggle ${isOpen ? 'hidden' : ''}`}
                onClick={onToggle}
                aria-label="Toggle menu"
            >
                <span className="hamburger">☰</span>
            </button>
            <div className={`sidebar ${isOpen ? 'open' : ''}`}>
                <div className="sidebar-header">
                    <h2>{t('navigation.stock')}</h2>
                    <button className="close-button" onClick={onToggle}>×</button>
                </div>
                <nav className="sidebar-nav">
                    <button 
                        className={`nav-item ${activeView === 'stock' ? 'active' : ''}`}
                        onClick={() => {
                            onViewChange('stock');
                            onToggle();
                        }}
                    >
                        <span className="icon">📈</span>
                        {t('navigation.stock')}
                    </button>
                    <button 
                        className={`nav-item ${activeView === 'hedge' ? 'active' : ''}`}
                        onClick={() => {
                            onViewChange('hedge');
                            onToggle();
                        }}
                    >
                        <span className="icon">🔄</span>
                        {t('navigation.hedge')}
                    </button>
                    <button 
                        className={`nav-item ${activeView === 'portfolio' ? 'active' : ''}`}
                        onClick={() => {
                            onViewChange('portfolio');
                            onToggle();
                        }}
                    >
                        <span className="icon">📊</span>
                        {t('navigation.portfolio')}
                    </button>
                    <button 
                        className={`nav-item ${activeView === 'financial' ? 'active' : ''}`}
                        onClick={() => {
                            onViewChange('financial');
                            onToggle();
                        }}
                    >
                        <span className="icon">📄</span>
                        {t('navigation.financial')}
                    </button>
                </nav>
            </div>
        </>
    );
};

export default Selector;
