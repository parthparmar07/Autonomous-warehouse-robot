// A small hand-rolled set of line icons (24x24, stroke-based, lucide-ish aesthetic)
// so the UI doesn't depend on a runtime icon library fighting React's reconciliation
// (lucide's DOM-mutating createIcons() swaps <i> for <svg> behind React's back when
// used via a plain <script> tag instead of the lucide-react package).

function Icon({ path, size = 16, strokeWidth = 2, ...rest }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" {...rest}>
      {path}
    </svg>
  );
}

const IconCube = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M12 2 3 7v10l9 5 9-5V7z" />
        <path d="M3 7l9 5 9-5" />
        <path d="M12 12v10" />
      </>
    }
  />
);

const IconGrid = (p) => (
  <Icon
    {...p}
    path={
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    }
  />
);

const IconChart = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M3 3v18h18" />
        <path d="M7 15l4-5 3 3 5-7" />
      </>
    }
  />
);

const IconTruck = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M2 8h11v9H2z" />
        <path d="M13 11h4l3 3v3h-7z" />
        <circle cx="6" cy="19" r="1.6" />
        <circle cx="16.5" cy="19" r="1.6" />
      </>
    }
  />
);

const IconLayers = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M12 3 2 8l10 5 10-5z" />
        <path d="M2 13l10 5 10-5" />
      </>
    }
  />
);

const IconSettings = (p) => (
  <Icon
    {...p}
    path={
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1 1.55V21a2 2 0 1 1-4 0v-.09A1.7 1.7 0 0 0 9 19.4a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.55-1H3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.55V3a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.7 1.7 0 0 0 19.4 9a1.7 1.7 0 0 0 1.55 1H21a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.51 1z" />
      </>
    }
  />
);

const IconSliders = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3" />
        <path d="M1 14h6M9 8h6M17 16h6" />
      </>
    }
  />
);

const IconFootprints = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M8 16.5c1.4 0 2.5-1.6 2.5-3.5S9.4 9.5 8 9.5 5.5 11.1 5.5 13s1.1 3.5 2.5 3.5z" />
        <path d="M16 21c1.4 0 2.5-1.6 2.5-3.5S17.4 14 16 14s-2.5 1.6-2.5 3.5S14.6 21 16 21z" />
        <path d="M7.5 9.8c.3-1 .1-2.3-.7-3-.9-.8-2 .2-1.6 1.4M15.5 14.3c.3-1 .1-2.3-.7-3-.9-.8-2 .2-1.6 1.4" />
      </>
    }
  />
);

const IconPackage = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M21 8 12 3 3 8v8l9 5 9-5z" />
        <path d="M3 8l9 5 9-5M12 13v8" />
      </>
    }
  />
);

const IconList = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M8 6h13M8 12h13M8 18h13" />
        <circle cx="3.5" cy="6" r="1" fill="currentColor" stroke="none" />
        <circle cx="3.5" cy="12" r="1" fill="currentColor" stroke="none" />
        <circle cx="3.5" cy="18" r="1" fill="currentColor" stroke="none" />
      </>
    }
  />
);

const IconAlertTriangle = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M10.3 3.9 1.8 18a1.5 1.5 0 0 0 1.3 2.3h17.8a1.5 1.5 0 0 0 1.3-2.3L13.7 3.9a1.5 1.5 0 0 0-2.6 0z" />
        <path d="M12 9v4M12 17h.01" />
      </>
    }
  />
);

const IconBan = (p) => (
  <Icon
    {...p}
    path={
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M5.5 5.5l13 13" />
      </>
    }
  />
);

const IconStar = (p) => (
  <Icon
    {...p}
    path={<path d="m12 2 3.1 6.3 6.9 1-5 4.9 1.2 6.9-6.2-3.3-6.2 3.3 1.2-6.9-5-4.9 6.9-1z" />}
  />
);

const IconActivity = (p) => (
  <Icon {...p} path={<path d="M22 12h-4l-3 8-6-16-3 8H2" />} />
);

const IconPlay = (p) => <Icon {...p} path={<path d="M6 3.5v17l14-8.5z" fill="currentColor" stroke="none" />} />;

const IconPause = (p) => (
  <Icon
    {...p}
    path={
      <>
        <rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor" stroke="none" />
        <rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor" stroke="none" />
      </>
    }
  />
);

const IconRotateCcw = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M3 12a9 9 0 1 0 3-6.7" />
        <path d="M3 4v5h5" />
      </>
    }
  />
);

const IconStepForward = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M5 4v16l11-8z" fill="currentColor" stroke="none" />
        <path d="M19 5v14" />
      </>
    }
  />
);

const IconRefreshCw = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M21 12a9 9 0 0 1-15.3 6.4L3 16" />
        <path d="M3 12a9 9 0 0 1 15.3-6.4L21 8" />
        <path d="M3 16v5h5M21 8V3h-5" />
      </>
    }
  />
);

const IconSearch = (p) => (
  <Icon
    {...p}
    path={
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m21 21-4.3-4.3" />
      </>
    }
  />
);

const IconMaximize = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" />
      </>
    }
  />
);

const IconChevronRight = (p) => <Icon {...p} path={<path d="m9 6 6 6-6 6" />} />;

const IconTrendUp = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="m3 17 6-6 4 4 8-8" />
        <path d="M17 7h4v4" />
      </>
    }
  />
);

const IconTrendDown = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="m3 7 6 6 4-4 8 8" />
        <path d="M17 17h4v-4" />
      </>
    }
  />
);

const IconMinus = (p) => <Icon {...p} path={<path d="M5 12h14" />} />;

const IconBox = (p) => (
  <Icon
    {...p}
    path={
      <>
        <path d="M21 8 12 3 3 8v8l9 5 9-5z" />
        <path d="M3 8l9 5 9-5" />
      </>
    }
  />
);
