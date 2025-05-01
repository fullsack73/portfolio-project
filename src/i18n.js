import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { loadTranslations } from './translationLoader';

// Initialize i18n
i18n
  .use(initReactI18next)
  .init({
    resources: {},
    lng: 'en', // default language
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    },
    ns: ['translation'],
    defaultNS: 'translation',
    react: {
      useSuspense: false
    }
  });

// Load translations after initialization
loadTranslations(i18n);

export default i18n;
