import enTranslation from '../locales/en/translation.json';
import koTranslation from '../locales/ko/translation.json';

export const loadTranslations = (i18n) => {
    i18n.addResourceBundle('en', 'translation', enTranslation, true, true);
    i18n.addResourceBundle('ko', 'translation', koTranslation, true, true);
};
